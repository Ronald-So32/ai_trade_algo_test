"""
Dynamic Stock Picker — Quantitative Universe Selection from Yahoo Finance
==========================================================================
Screens all publicly tradable US stocks using factor-based criteria to build
an optimal universe for the existing strategy suite.

Academic basis:
  - Fama & French (1993, 2015): Factor-based screening
  - Pastor & Stambaugh (2003): Liquidity as a priced factor
  - Amihud (2002): Illiquidity measure for universe filtering
  - Ang, Hodrick, Xing & Zhang (2006): Volatility characteristics

The picker is **dynamic**: it can be re-run at any frequency (e.g. monthly)
to refresh the universe based on current market conditions, strategy fit,
and regime state.  This makes it suitable for both backtesting and live
paper-trading.

Usage:
    picker = StockPicker(config)
    universe_df = picker.pick(n_stocks=100)
    tickers = universe_df['ticker'].tolist()
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Sector ETF proxies for correlation/beta estimation
# ──────────────────────────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# ──────────────────────────────────────────────────────────────────────────────
# Universe sourcing: S&P 500 + NASDAQ-100 + Russell 1000 proxy
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            header=0,
        )
        df = tables[0]
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = df[col].str.replace(".", "-", regex=False).tolist()
        return [t.strip() for t in tickers if isinstance(t, str) and len(t) <= 5]
    except Exception as e:
        logger.warning("Failed to fetch S&P 500 list: %s", e)
        return []


def _fetch_nasdaq100_tickers() -> list[str]:
    """Fetch NASDAQ-100 constituents from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            header=0,
        )
        for tbl in tables:
            if "Ticker" in tbl.columns:
                return tbl["Ticker"].str.strip().tolist()
            if "Symbol" in tbl.columns:
                return tbl["Symbol"].str.strip().tolist()
        return []
    except Exception as e:
        logger.warning("Failed to fetch NASDAQ-100 list: %s", e)
        return []


def get_all_us_tickers() -> list[str]:
    """
    Get a broad list of US-traded tickers by combining S&P 500 and NASDAQ-100.
    Deduplicates and returns sorted list.
    """
    sp500 = _fetch_sp500_tickers()
    ndx100 = _fetch_nasdaq100_tickers()
    all_tickers = sorted(set(sp500 + ndx100))
    logger.info(
        "Sourced %d unique tickers (S&P500=%d, NASDAQ100=%d)",
        len(all_tickers), len(sp500), len(ndx100),
    )
    return all_tickers


class StockPicker:
    """
    Quantitative stock screener that selects an optimal universe for
    the existing strategy suite.

    Screening criteria (configurable):
      1. Market cap > min_market_cap (liquidity, Pastor & Stambaugh 2003)
      2. Avg daily dollar volume > min_dollar_volume (tradability)
      3. Price > min_price (avoid penny stocks)
      4. History length >= min_history_days
      5. Volatility within [vol_floor, vol_cap] (strategy-appropriate range)
      6. Sector diversification (max_sector_pct)
      7. Mean-reversion score (Hurst exponent proxy) — for pairs/MR strategies
      8. Momentum quality score — for trend strategies
      9. Beta diversity — ensure beta spread for BAB strategy

    Parameters
    ----------
    min_market_cap : float
        Minimum market cap in USD (default $500M).
    min_dollar_volume : float
        Minimum average daily dollar volume (default $5M).
    min_price : float
        Minimum stock price (default $5).
    min_history_days : int
        Minimum trading days of history (default 504 = ~2 years).
    vol_floor : float
        Minimum annualized volatility (default 0.10).
    vol_cap : float
        Maximum annualized volatility (default 0.80).
    max_sector_pct : float
        Maximum fraction of universe from one sector (default 0.30).
    lookback_days : int
        Days of history to download for screening (default 756 = 3 years).
    """

    def __init__(
        self,
        min_market_cap: float = 5e8,
        min_dollar_volume: float = 5e6,
        min_price: float = 5.0,
        min_history_days: int = 504,
        vol_floor: float = 0.10,
        vol_cap: float = 0.80,
        max_sector_pct: float = 0.30,
        lookback_days: int = 756,
    ) -> None:
        self.min_market_cap = min_market_cap
        self.min_dollar_volume = min_dollar_volume
        self.min_price = min_price
        self.min_history_days = min_history_days
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap
        self.max_sector_pct = max_sector_pct
        self.lookback_days = lookback_days

    def pick(
        self,
        n_stocks: int = 100,
        candidate_tickers: Optional[list[str]] = None,
        regime: str = "normal",
    ) -> pd.DataFrame:
        """
        Screen and rank stocks, returning the top n_stocks.

        Parameters
        ----------
        n_stocks : int
            Target universe size.
        candidate_tickers : list[str], optional
            Pre-defined ticker list to screen from. If None, fetches
            S&P500 + NASDAQ-100 automatically.
        regime : str
            Current market regime ("bull", "normal", "crisis").
            Adjusts screening criteria dynamically.

        Returns
        -------
        pd.DataFrame
            Selected stocks with columns: ticker, sector, market_cap,
            dollar_volume, volatility, momentum_score, mr_score,
            beta, composite_score.
        """
        if candidate_tickers is None:
            candidate_tickers = get_all_us_tickers()

        if not candidate_tickers:
            raise ValueError("No candidate tickers available for screening.")

        logger.info("Screening %d candidates for top %d stocks...", len(candidate_tickers), n_stocks)

        # Step 1: Fetch basic info and price data
        screened = self._fetch_and_screen(candidate_tickers)

        if screened.empty:
            raise ValueError("No stocks passed screening criteria.")

        # Step 2: Compute composite score
        screened = self._compute_scores(screened, regime=regime)

        # Step 3: Sector diversification enforcement
        selected = self._enforce_sector_balance(screened, n_stocks)

        logger.info(
            "Selected %d stocks across %d sectors (top composite score: %.3f)",
            len(selected), selected["sector"].nunique(),
            selected["composite_score"].max(),
        )

        return selected.reset_index(drop=True)

    def _fetch_and_screen(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch data and apply hard screening filters."""
        rows = []
        batch_size = 20

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            for ticker in batch:
                try:
                    info = self._get_ticker_info(ticker)
                    if info is not None:
                        rows.append(info)
                except Exception:
                    continue

            if i + batch_size < len(tickers):
                time.sleep(0.5)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(
                    "  Screened %d/%d tickers (%d passed so far)",
                    min(i + batch_size, len(tickers)), len(tickers), len(rows),
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info("  %d / %d tickers passed hard screening filters", len(df), len(tickers))
        return df

    def _get_ticker_info(self, ticker: str) -> Optional[dict]:
        """Fetch info for a single ticker and apply hard filters."""
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info

            # Market cap filter
            mkt_cap = getattr(info, "market_cap", None) or 0
            if mkt_cap < self.min_market_cap:
                return None

            # Price filter
            last_price = getattr(info, "last_price", None) or 0
            if last_price < self.min_price:
                return None

            # Get sector from full info (cached by yfinance)
            try:
                full_info = t.info
                sector = full_info.get("sector", "Unknown")
                industry = full_info.get("industry", "Unknown")
                beta = full_info.get("beta", 1.0) or 1.0
            except Exception:
                sector = "Unknown"
                industry = "Unknown"
                beta = 1.0

            # Fetch recent history for volume and volatility checks
            hist = t.history(period=f"{self.lookback_days // 252 + 1}y", auto_adjust=True)
            if hist.empty or len(hist) < self.min_history_days:
                return None

            hist.index = hist.index.tz_localize(None)

            # Dollar volume filter
            avg_dollar_vol = (hist["Close"] * hist["Volume"]).tail(63).mean()
            if avg_dollar_vol < self.min_dollar_volume:
                return None

            # Volatility filter
            returns = hist["Close"].pct_change().dropna()
            ann_vol = returns.std() * np.sqrt(252)
            if ann_vol < self.vol_floor or ann_vol > self.vol_cap:
                return None

            # Compute additional metrics for scoring
            # Momentum: 12-1 month return (skip most recent month)
            if len(hist) >= 252:
                mom_12m = (hist["Close"].iloc[-21] / hist["Close"].iloc[-252]) - 1
            else:
                mom_12m = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1

            # Mean-reversion score: negative autocorrelation of returns
            # (proxy for Hurst exponent < 0.5)
            if len(returns) >= 63:
                ac1 = returns.tail(252).autocorr(lag=1)
                mr_score = -ac1 if not np.isnan(ac1) else 0.0
            else:
                mr_score = 0.0

            return {
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "market_cap": mkt_cap,
                "last_price": last_price,
                "dollar_volume": avg_dollar_vol,
                "volatility": ann_vol,
                "momentum_score": mom_12m,
                "mr_score": mr_score,
                "beta": beta,
                "n_days": len(hist),
            }

        except Exception:
            return None

    def _compute_scores(self, df: pd.DataFrame, regime: str = "normal") -> pd.DataFrame:
        """
        Compute composite score for ranking.

        Score components (research-backed weights):
          - Liquidity (30%): log dollar volume rank — Pastor & Stambaugh (2003)
          - Volatility suitability (20%): penalize extremes — Ang et al. (2006)
          - Momentum quality (20%): 12-1m return rank — Jegadeesh & Titman (1993)
          - Mean-reversion (15%): autocorrelation score — Lo & MacKinlay (1990)
          - Beta diversity (15%): spread from 1.0 — Frazzini & Pedersen (2014)
        """
        out = df.copy()

        # Rank-based scoring (0 to 1)
        n = len(out)
        if n == 0:
            return out

        # Liquidity rank (higher is better)
        out["liq_rank"] = out["dollar_volume"].rank(pct=True)

        # Volatility suitability: prefer moderate vol (0.15–0.40 annualized)
        ideal_vol = 0.25
        out["vol_suit"] = 1.0 - (out["volatility"] - ideal_vol).abs() / 0.40
        out["vol_suit"] = out["vol_suit"].clip(0, 1)

        # Momentum rank (higher is better)
        out["mom_rank"] = out["momentum_score"].rank(pct=True)

        # Mean-reversion rank (higher mr_score is better for pairs/MR)
        out["mr_rank"] = out["mr_score"].rank(pct=True)

        # Beta diversity: prefer stocks away from beta=1 (for BAB strategy)
        out["beta_div"] = (out["beta"] - 1.0).abs().rank(pct=True)

        # Regime-adaptive weighting
        if regime == "crisis":
            # In crisis: favor low vol, high liquidity, less momentum
            weights = {"liq_rank": 0.40, "vol_suit": 0.25, "mom_rank": 0.10,
                       "mr_rank": 0.15, "beta_div": 0.10}
        elif regime == "bull":
            # In bull: favor momentum, moderate liquidity
            weights = {"liq_rank": 0.25, "vol_suit": 0.15, "mom_rank": 0.30,
                       "mr_rank": 0.15, "beta_div": 0.15}
        else:
            # Normal: balanced
            weights = {"liq_rank": 0.30, "vol_suit": 0.20, "mom_rank": 0.20,
                       "mr_rank": 0.15, "beta_div": 0.15}

        out["composite_score"] = sum(
            out[col] * w for col, w in weights.items()
        )

        return out.sort_values("composite_score", ascending=False)

    def _enforce_sector_balance(
        self, df: pd.DataFrame, n_stocks: int
    ) -> pd.DataFrame:
        """
        Select top stocks while enforcing sector diversification.

        Uses a greedy algorithm: iterate through ranked stocks, adding
        each one unless its sector is already at the cap.
        """
        max_per_sector = max(1, int(n_stocks * self.max_sector_pct))
        sector_counts: dict[str, int] = {}
        selected_idx = []

        for idx, row in df.iterrows():
            sector = row["sector"]
            current = sector_counts.get(sector, 0)
            if current < max_per_sector:
                selected_idx.append(idx)
                sector_counts[sector] = current + 1

            if len(selected_idx) >= n_stocks:
                break

        return df.loc[selected_idx]

    def pick_for_strategies(
        self,
        n_stocks: int = 100,
        candidate_tickers: Optional[list[str]] = None,
        crisis_prob: float = 0.0,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Pick stocks and additionally group them by strategy suitability.

        Returns
        -------
        (universe_df, strategy_groups)
            strategy_groups maps strategy name to list of tickers
            particularly suited for that strategy.
        """
        regime = "crisis" if crisis_prob > 0.5 else ("bull" if crisis_prob < 0.2 else "normal")
        universe = self.pick(n_stocks=n_stocks, candidate_tickers=candidate_tickers, regime=regime)

        # Group by strategy suitability
        groups: dict[str, list[str]] = {}

        # Momentum strategies: high momentum_score, moderate vol
        mom_mask = (
            (universe["momentum_score"] > universe["momentum_score"].median()) &
            (universe["volatility"] < universe["volatility"].quantile(0.75))
        )
        groups["momentum"] = universe.loc[mom_mask, "ticker"].tolist()

        # Mean-reversion / pairs: high mr_score, lower vol
        mr_mask = (
            (universe["mr_score"] > universe["mr_score"].median()) &
            (universe["volatility"] < universe["volatility"].quantile(0.80))
        )
        groups["mean_reversion"] = universe.loc[mr_mask, "ticker"].tolist()

        # BAB: extreme betas (both high and low)
        beta_low = universe["beta"] < universe["beta"].quantile(0.25)
        beta_high = universe["beta"] > universe["beta"].quantile(0.75)
        groups["bab"] = universe.loc[beta_low | beta_high, "ticker"].tolist()

        # Vol breakout: higher vol stocks
        vol_mask = universe["volatility"] > universe["volatility"].median()
        groups["volatility"] = universe.loc[vol_mask, "ticker"].tolist()

        # All stocks for general strategies
        groups["all"] = universe["ticker"].tolist()

        for name, tickers in groups.items():
            logger.info("  Strategy group '%s': %d stocks", name, len(tickers))

        return universe, groups
