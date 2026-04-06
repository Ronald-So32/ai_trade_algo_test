"""
Real Market Data Fetcher — downloads historical OHLCV data from Yahoo Finance
for ~150 US large-cap equities across 11 GICS sectors.

Expanded universe (v3.1) rationale:
  - Cross-sectional strategies (Residual Reversal, 52-Week High) need large universes
    for reliable quintile sorts. With 49 stocks: 10 per quintile (noisy).
    With 150 stocks: 30 per quintile (robust). Blitz et al. (2013) used full CRSP (~3000).
  - More stocks = more sub-industry groups for finer residual computation
  - All stocks: S&P 500 large-caps, $10B+ market cap, highly liquid
  - Sub-industry classifications enable Blitz et al. (2023) within-industry reversal

Produces DataFrames matching the exact schema expected by the existing pipeline:
  - security_master: security_id, ticker, cusip, isin, company_name, sector, ...
  - market_data: date, security_id, open, high, low, close, adjusted_close, volume, ...
  - returns: computed by ReturnsCalculator (unchanged)
  - universes: computed by UniverseConstructor (unchanged)

Usage:
    fetcher = RealDataFetcher(config=config)
    security_master, market_data = fetcher.fetch()
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from qrt.utils.config import Config
from qrt.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Universe definition: ~150 S&P 500 large-caps across 11 GICS sectors
#
# Selection criteria (all survivorship-bias-free S&P 500 members):
#   - Market cap > $10B (highly liquid, tradable on Alpaca)
#   - History back to at least 2010 (full backtest coverage)
#   - Sector-balanced: no sector > 25% of universe
#   - Emphasis on sub-industry diversity for residual reversal signal quality
# ──────────────────────────────────────────────────────────────────────────────
REAL_UNIVERSE: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC",
        "AMAT", "LRCX", "KLAC", "MCHP", "ADI", "SNPS", "CDNS",
        "CRM", "ADBE", "ORCL", "NOW", "INTU", "ACN", "IBM",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP",
        "BLK", "SCHW", "USB", "PNC", "TFC", "AIG", "MET",
        "PRU", "CB", "ICE", "CME", "BK",
    ],
    "Healthcare": [
        "LLY", "JNJ", "MRK", "ABBV", "PFE", "TMO", "DHR",
        "UNH", "ABT", "BMY", "AMGN", "GILD", "MDT", "SYK",
        "ISRG", "BSX", "ELV", "CI", "HUM", "ZTS",
    ],
    "Consumer Discretionary": [
        "AMZN", "HD", "LOW", "MCD", "NKE", "SBUX", "TJX",
        "BKNG", "MAR", "GM", "F", "ORLY", "ROST", "DHI",
        "LEN",
    ],
    "Consumer Staples": [
        "COST", "WMT", "KO", "PEP", "PG", "PM", "MO",
        "CL", "KMB", "GIS", "SYY", "KR", "STZ",
    ],
    "Industrials": [
        "CAT", "DE", "RTX", "GE", "HON", "UNP", "UPS",
        "BA", "LMT", "NOC", "GD", "WM", "EMR", "ETN",
        "ITW", "FDX", "CSX", "NSC",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC",
        "PSX", "VLO", "DVN", "HAL",
    ],
    "Communications": [
        "GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "CMCSA",
        "T", "CHTR", "EA", "TTWO",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC",
    ],
    "Real Estate": [
        "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE",
    ],
}

# Company names for the security master
COMPANY_NAMES: dict[str, str] = {
    # Technology
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "AVGO": "Broadcom Inc.", "AMD": "Advanced Micro Devices Inc.", "QCOM": "QUALCOMM Inc.",
    "TXN": "Texas Instruments Inc.", "INTC": "Intel Corporation",
    "AMAT": "Applied Materials Inc.", "LRCX": "Lam Research Corp.", "KLAC": "KLA Corporation",
    "MCHP": "Microchip Technology Inc.", "ADI": "Analog Devices Inc.",
    "SNPS": "Synopsys Inc.", "CDNS": "Cadence Design Systems Inc.",
    "CRM": "Salesforce Inc.", "ADBE": "Adobe Inc.", "ORCL": "Oracle Corporation",
    "NOW": "ServiceNow Inc.", "INTU": "Intuit Inc.", "ACN": "Accenture plc", "IBM": "IBM Corporation",
    # Financials
    "JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corp.",
    "WFC": "Wells Fargo & Co.", "C": "Citigroup Inc.", "GS": "Goldman Sachs Group Inc.",
    "MS": "Morgan Stanley", "AXP": "American Express Co.",
    "BLK": "BlackRock Inc.", "SCHW": "Charles Schwab Corp.", "USB": "U.S. Bancorp",
    "PNC": "PNC Financial Services Group", "TFC": "Truist Financial Corp.",
    "AIG": "American International Group", "MET": "MetLife Inc.",
    "PRU": "Prudential Financial Inc.", "CB": "Chubb Limited",
    "ICE": "Intercontinental Exchange Inc.", "CME": "CME Group Inc.",
    "BK": "Bank of New York Mellon Corp.", "MMC": "Marsh & McLennan Cos.",
    # Healthcare
    "LLY": "Eli Lilly and Co.", "JNJ": "Johnson & Johnson", "MRK": "Merck & Co. Inc.",
    "ABBV": "AbbVie Inc.", "PFE": "Pfizer Inc.", "TMO": "Thermo Fisher Scientific Inc.",
    "DHR": "Danaher Corporation", "UNH": "UnitedHealth Group Inc.", "ABT": "Abbott Laboratories",
    "BMY": "Bristol-Myers Squibb Co.", "AMGN": "Amgen Inc.", "GILD": "Gilead Sciences Inc.",
    "MDT": "Medtronic plc", "SYK": "Stryker Corporation", "ISRG": "Intuitive Surgical Inc.",
    "BSX": "Boston Scientific Corp.", "ELV": "Elevance Health Inc.", "CI": "Cigna Group",
    "HUM": "Humana Inc.", "ZTS": "Zoetis Inc.",
    # Consumer Discretionary
    "AMZN": "Amazon.com Inc.", "HD": "Home Depot Inc.", "LOW": "Lowe's Companies Inc.",
    "MCD": "McDonald's Corp.", "NKE": "Nike Inc.", "SBUX": "Starbucks Corp.",
    "TJX": "TJX Companies Inc.", "BKNG": "Booking Holdings Inc.",
    "MAR": "Marriott International Inc.", "GM": "General Motors Co.", "F": "Ford Motor Co.",
    "ORLY": "O'Reilly Automotive Inc.", "ROST": "Ross Stores Inc.",
    "DHI": "D.R. Horton Inc.", "LEN": "Lennar Corporation",
    # Consumer Staples
    "COST": "Costco Wholesale Corp.", "WMT": "Walmart Inc.", "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.", "PG": "Procter & Gamble Co.", "PM": "Philip Morris International",
    "MO": "Altria Group Inc.", "CL": "Colgate-Palmolive Co.", "KMB": "Kimberly-Clark Corp.",
    "GIS": "General Mills Inc.", "SYY": "Sysco Corporation", "KR": "Kroger Co.",
    "STZ": "Constellation Brands Inc.",
    # Industrials
    "CAT": "Caterpillar Inc.", "DE": "Deere & Company", "RTX": "RTX Corporation",
    "GE": "General Electric Co.", "HON": "Honeywell International Inc.",
    "UNP": "Union Pacific Corp.", "UPS": "United Parcel Service Inc.",
    "BA": "Boeing Company", "LMT": "Lockheed Martin Corp.", "NOC": "Northrop Grumman Corp.",
    "GD": "General Dynamics Corp.", "WM": "Waste Management Inc.", "EMR": "Emerson Electric Co.",
    "ETN": "Eaton Corporation plc", "ITW": "Illinois Tool Works Inc.",
    "FDX": "FedEx Corporation", "CSX": "CSX Corporation", "NSC": "Norfolk Southern Corp.",
    # Energy
    "XOM": "Exxon Mobil Corp.", "CVX": "Chevron Corporation", "COP": "ConocoPhillips",
    "EOG": "EOG Resources Inc.", "SLB": "Schlumberger Limited",
    "OXY": "Occidental Petroleum Corp.", "MPC": "Marathon Petroleum Corp.",
    "PSX": "Phillips 66", "VLO": "Valero Energy Corp.",
    "DVN": "Devon Energy Corp.", "HAL": "Halliburton Co.",
    # Communications
    "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.", "NFLX": "Netflix Inc.",
    "DIS": "Walt Disney Co.", "TMUS": "T-Mobile US Inc.", "VZ": "Verizon Communications Inc.",
    "CMCSA": "Comcast Corporation", "T": "AT&T Inc.", "CHTR": "Charter Communications Inc.",
    "EA": "Electronic Arts Inc.", "TTWO": "Take-Two Interactive Software Inc.",
    # Utilities
    "NEE": "NextEra Energy Inc.", "DUK": "Duke Energy Corp.", "SO": "Southern Company",
    "D": "Dominion Energy Inc.", "AEP": "American Electric Power Co.",
    "SRE": "Sempra", "EXC": "Exelon Corporation",
    # Real Estate
    "PLD": "Prologis Inc.", "AMT": "American Tower Corp.", "CCI": "Crown Castle Inc.",
    "EQIX": "Equinix Inc.", "SPG": "Simon Property Group Inc.",
    "PSA": "Public Storage", "O": "Realty Income Corp.",
    # Materials
    "LIN": "Linde plc", "APD": "Air Products & Chemicals Inc.",
    "SHW": "Sherwin-Williams Co.", "ECL": "Ecolab Inc.", "NEM": "Newmont Corp.",
    "FCX": "Freeport-McMoRan Inc.", "NUE": "Nucor Corporation",
}

EXCHANGE_MAP: dict[str, str] = {
    "AAPL": "NASDAQ", "MSFT": "NASDAQ", "NVDA": "NASDAQ", "AVGO": "NASDAQ",
    "AMD": "NASDAQ", "QCOM": "NASDAQ", "TXN": "NASDAQ", "INTC": "NASDAQ",
    "AMAT": "NASDAQ", "LRCX": "NASDAQ", "KLAC": "NASDAQ", "MCHP": "NASDAQ",
    "ADI": "NASDAQ", "SNPS": "NASDAQ", "CDNS": "NASDAQ", "CRM": "NYSE",
    "ADBE": "NASDAQ", "ORCL": "NYSE", "NOW": "NYSE", "INTU": "NASDAQ",
    "GOOGL": "NASDAQ", "META": "NASDAQ", "NFLX": "NASDAQ", "TMUS": "NASDAQ",
    "CMCSA": "NASDAQ", "AMZN": "NASDAQ", "COST": "NASDAQ", "SBUX": "NASDAQ",
    "BKNG": "NASDAQ", "ORLY": "NASDAQ", "ROST": "NASDAQ", "GILD": "NASDAQ",
    "AMGN": "NASDAQ", "ISRG": "NASDAQ", "EA": "NASDAQ", "TTWO": "NASDAQ",
    "CHTR": "NASDAQ", "CSX": "NASDAQ",
}

# Industry sub-classifications (GICS sub-industry level)
# Finer granularity = better within-industry residuals for Blitz et al. (2023)
INDUSTRY_MAP: dict[str, str] = {
    # Technology — Semiconductors
    "NVDA": "Semiconductors", "AVGO": "Semiconductors", "AMD": "Semiconductors",
    "QCOM": "Semiconductors", "TXN": "Semiconductors", "INTC": "Semiconductors",
    "AMAT": "Semiconductor Equipment", "LRCX": "Semiconductor Equipment",
    "KLAC": "Semiconductor Equipment", "MCHP": "Semiconductors", "ADI": "Semiconductors",
    # Technology — Software & IT Services
    "AAPL": "Consumer Electronics", "MSFT": "Software",
    "SNPS": "EDA Software", "CDNS": "EDA Software",
    "CRM": "Enterprise Software", "ADBE": "Enterprise Software", "ORCL": "Enterprise Software",
    "NOW": "Enterprise Software", "INTU": "Enterprise Software",
    "ACN": "IT Consulting", "IBM": "IT Consulting",
    # Financials — Banks
    "JPM": "Diversified Banks", "BAC": "Diversified Banks", "WFC": "Diversified Banks",
    "C": "Diversified Banks", "USB": "Regional Banks", "PNC": "Regional Banks",
    "TFC": "Regional Banks", "BK": "Custody Banks",
    # Financials — Capital Markets & Insurance
    "GS": "Investment Banking", "MS": "Investment Banking",
    "BLK": "Asset Management", "SCHW": "Brokerage",
    "ICE": "Financial Exchanges", "CME": "Financial Exchanges",
    "AXP": "Consumer Finance", "MMC": "Insurance Brokerage",
    "AIG": "Property & Casualty Insurance", "MET": "Life Insurance",
    "PRU": "Life Insurance", "CB": "Property & Casualty Insurance",
    # Healthcare — Pharma & Biotech
    "LLY": "Pharmaceuticals", "JNJ": "Pharmaceuticals", "MRK": "Pharmaceuticals",
    "ABBV": "Pharmaceuticals", "PFE": "Pharmaceuticals", "BMY": "Pharmaceuticals",
    "AMGN": "Biotechnology", "GILD": "Biotechnology", "ZTS": "Veterinary Pharma",
    # Healthcare — Devices & Services
    "TMO": "Life Sciences Tools", "DHR": "Life Sciences Tools", "ABT": "Medical Devices",
    "MDT": "Medical Devices", "SYK": "Medical Devices", "ISRG": "Medical Devices",
    "BSX": "Medical Devices",
    "UNH": "Managed Care", "ELV": "Managed Care", "CI": "Managed Care", "HUM": "Managed Care",
    # Consumer Discretionary
    "AMZN": "Internet Retail", "HD": "Home Improvement", "LOW": "Home Improvement",
    "MCD": "Restaurants", "SBUX": "Restaurants", "NKE": "Apparel",
    "TJX": "Discount Retail", "ROST": "Discount Retail",
    "BKNG": "Online Travel", "MAR": "Hotels",
    "GM": "Automobiles", "F": "Automobiles",
    "ORLY": "Auto Parts", "DHI": "Homebuilders", "LEN": "Homebuilders",
    # Consumer Staples
    "COST": "Warehouse Clubs", "WMT": "Discount Stores", "KR": "Food Retail",
    "KO": "Beverages", "PEP": "Beverages", "STZ": "Beverages",
    "PG": "Household Products", "CL": "Household Products", "KMB": "Household Products",
    "PM": "Tobacco", "MO": "Tobacco",
    "GIS": "Packaged Foods", "SYY": "Food Distribution",
    # Industrials — Aerospace & Defense
    "BA": "Aerospace & Defense", "RTX": "Aerospace & Defense",
    "LMT": "Aerospace & Defense", "NOC": "Aerospace & Defense", "GD": "Aerospace & Defense",
    # Industrials — Machinery & Transport
    "CAT": "Farm & Heavy Equipment", "DE": "Farm & Heavy Equipment",
    "GE": "Diversified Industrials", "HON": "Diversified Industrials",
    "EMR": "Electrical Equipment", "ETN": "Electrical Equipment", "ITW": "Specialty Industrials",
    "UNP": "Railroads", "CSX": "Railroads", "NSC": "Railroads",
    "UPS": "Logistics", "FDX": "Logistics", "WM": "Waste Management",
    # Energy
    "XOM": "Oil & Gas Integrated", "CVX": "Oil & Gas Integrated",
    "COP": "Oil & Gas E&P", "EOG": "Oil & Gas E&P",
    "DVN": "Oil & Gas E&P", "OXY": "Oil & Gas E&P",
    "SLB": "Oil & Gas Services", "HAL": "Oil & Gas Services",
    "MPC": "Oil & Gas Refining", "PSX": "Oil & Gas Refining", "VLO": "Oil & Gas Refining",
    # Communications
    "GOOGL": "Internet Services", "META": "Internet Services",
    "NFLX": "Entertainment", "DIS": "Entertainment",
    "EA": "Video Games", "TTWO": "Video Games",
    "TMUS": "Telecom", "VZ": "Telecom", "T": "Telecom",
    "CMCSA": "Cable & Satellite", "CHTR": "Cable & Satellite",
    # Utilities
    "NEE": "Electric Utilities", "DUK": "Electric Utilities", "SO": "Electric Utilities",
    "D": "Electric Utilities", "AEP": "Electric Utilities",
    "SRE": "Multi-Utilities", "EXC": "Electric Utilities",
    # Real Estate
    "PLD": "Industrial REITs", "AMT": "Telecom REITs", "CCI": "Telecom REITs",
    "EQIX": "Data Center REITs", "SPG": "Retail REITs",
    "PSA": "Storage REITs", "O": "Net Lease REITs",
    # Materials
    "LIN": "Industrial Gases", "APD": "Industrial Gases",
    "SHW": "Specialty Chemicals", "ECL": "Specialty Chemicals",
    "NEM": "Gold Mining", "FCX": "Copper Mining", "NUE": "Steel",
}


def _ticker_to_cusip(ticker: str) -> str:
    """Generate a deterministic pseudo-CUSIP from ticker (for schema compliance)."""
    h = hashlib.md5(ticker.encode()).hexdigest()[:9].upper()
    return h


class RealDataFetcher:
    """
    Fetch real historical market data from Yahoo Finance for ~150 US large-cap equities.

    Parameters
    ----------
    config : Config, optional
    start_date : str
        Start date for historical data (default: 2010-01-01).
    end_date : str or None
        End date (default: today).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> None:
        self._config = config or Config()
        self._start_date = start_date
        self._end_date = end_date
        self._all_tickers: list[str] = []
        self._ticker_sector: dict[str, str] = {}

        for sector, tickers in REAL_UNIVERSE.items():
            for t in tickers:
                self._all_tickers.append(t)
                self._ticker_sector[t] = sector

    def fetch(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download data and return (security_master, market_data) DataFrames
        matching the existing pipeline schema.
        """
        logger.info("Fetching real market data for %d tickers …", len(self._all_tickers))
        logger.info("Date range: %s → %s", self._start_date, self._end_date or "today")

        # ── Step 1: Download OHLCV via yfinance ──────────────────────────
        raw = self._download_all()

        # ── Step 2: Build security master ────────────────────────────────
        security_master = self._build_security_master(raw)

        # ── Step 3: Build market data in canonical long format ───────────
        market_data = self._build_market_data(raw, security_master)

        return security_master, market_data

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_all(self) -> dict[str, pd.DataFrame]:
        """Download OHLCV + dividends + splits for all tickers."""
        result: dict[str, pd.DataFrame] = {}
        batch_size = 10

        for i in range(0, len(self._all_tickers), batch_size):
            batch = self._all_tickers[i : i + batch_size]
            logger.info(
                "  Downloading batch %d/%d: %s",
                i // batch_size + 1,
                (len(self._all_tickers) + batch_size - 1) // batch_size,
                ", ".join(batch),
            )

            for ticker in batch:
                try:
                    yf_ticker = yf.Ticker(ticker)
                    hist = yf_ticker.history(
                        start=self._start_date,
                        end=self._end_date,
                        auto_adjust=False,
                        actions=True,
                    )
                    if hist.empty:
                        logger.warning("  No data for %s — skipping", ticker)
                        continue

                    hist.index = hist.index.tz_localize(None)
                    result[ticker] = hist
                    logger.info("    %s: %d rows (%s → %s)",
                                ticker, len(hist),
                                str(hist.index.min().date()),
                                str(hist.index.max().date()))
                except Exception as e:
                    logger.warning("  Failed to download %s: %s", ticker, e)

            # Be polite to Yahoo Finance
            if i + batch_size < len(self._all_tickers):
                time.sleep(1)

        logger.info("Downloaded data for %d / %d tickers", len(result), len(self._all_tickers))
        return result

    # ------------------------------------------------------------------
    # Security Master
    # ------------------------------------------------------------------

    def _build_security_master(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build security_master DataFrame matching synthetic schema."""
        rows = []
        for sid, ticker in enumerate(sorted(raw.keys()), start=1):
            hist = raw[ticker]
            sector = self._ticker_sector[ticker]
            cusip = _ticker_to_cusip(ticker)
            isin = f"US{cusip}"

            rows.append({
                "security_id": sid,
                "ticker": ticker,
                "cusip": cusip,
                "isin": isin,
                "company_name": COMPANY_NAMES.get(ticker, ticker),
                "sector": sector,
                "industry": INDUSTRY_MAP.get(ticker, sector),
                "exchange": EXCHANGE_MAP.get(ticker, "NYSE"),
                "currency": "USD",
                "country": "US",
                "list_date": hist.index.min(),
                "delist_date": pd.NaT,
                "is_active": True,
            })

        sm = pd.DataFrame(rows)
        sm["list_date"] = pd.to_datetime(sm["list_date"])
        logger.info("Security master: %d securities across %d sectors",
                     len(sm), sm["sector"].nunique())
        return sm

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    def _build_market_data(
        self, raw: dict[str, pd.DataFrame], security_master: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert raw yfinance DataFrames to canonical long-format market_data
        matching the synthetic schema exactly.
        """
        ticker_to_sid = dict(zip(security_master["ticker"], security_master["security_id"]))
        frames: list[pd.DataFrame] = []

        for ticker, hist in raw.items():
            sid = ticker_to_sid[ticker]
            n = len(hist)

            # Yahoo provides: Open, High, Low, Close, Adj Close, Volume,
            #                 Dividends, Stock Splits
            close = hist["Close"].values.astype(float)
            adj_close = hist["Adj Close"].values.astype(float)
            volume = hist["Volume"].values.astype(float)
            high = hist["High"].values.astype(float)
            low = hist["Low"].values.astype(float)

            # VWAP approximation: (High + Low + Close) / 3
            vwap = (high + low + close) / 3.0

            # Dollar volume
            dollar_volume = vwap * volume

            # Dividends and splits from yfinance
            dividends = hist["Dividends"].values.astype(float) if "Dividends" in hist.columns else np.zeros(n)
            splits = hist["Stock Splits"].values.astype(float) if "Stock Splits" in hist.columns else np.ones(n)
            # Convert 0.0 splits to 1.0 (yfinance uses 0.0 for no-split days)
            splits = np.where(splits == 0.0, 1.0, splits)

            # Market cap estimate: use adj_close * a rough shares outstanding
            # We'll estimate from the most recent data point
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.fast_info
                shares_out = getattr(info, "shares", None)
                if shares_out is None or shares_out == 0:
                    # Fallback: estimate from market cap / price
                    mkt_cap_latest = getattr(info, "market_cap", None)
                    if mkt_cap_latest and adj_close[-1] > 0:
                        shares_out = mkt_cap_latest / adj_close[-1]
                    else:
                        shares_out = 1e9 / adj_close[-1] if adj_close[-1] > 0 else 1e7
            except Exception:
                shares_out = 1e9 / adj_close[-1] if adj_close[-1] > 0 else 1e7

            market_cap = adj_close * float(shares_out)

            frame = pd.DataFrame({
                "date": hist.index,
                "security_id": sid,
                "open": hist["Open"].values.astype(float),
                "high": high,
                "low": low,
                "close": close,
                "adjusted_close": adj_close,
                "volume": volume.astype(np.int64),
                "vwap": vwap,
                "dollar_volume": dollar_volume,
                "market_cap": market_cap,
                "split_factor": splits,
                "dividend_amount": dividends,
            })
            frames.append(frame)

        md = pd.concat(frames, ignore_index=True)
        md["date"] = pd.to_datetime(md["date"])
        md = md.sort_values(["date", "security_id"]).reset_index(drop=True)

        logger.info(
            "Market data: %d rows, %d securities, %s → %s",
            len(md), md["security_id"].nunique(),
            str(md["date"].min().date()), str(md["date"].max().date()),
        )

        # ── Data quality checks ──────────────────────────────────────────
        n_neg = (md["close"] < 0).sum()
        if n_neg > 0:
            logger.warning("Found %d negative close prices — clipping to 0.01", n_neg)
            md.loc[md["close"] < 0, "close"] = 0.01

        n_zero_vol = (md["volume"] <= 0).sum()
        if n_zero_vol > 0:
            logger.warning("Found %d zero-volume rows", n_zero_vol)

        return md
