"""
Security Master — synthetic but realistic US equity universe.

Covers 150+ securities across major GICS-style sectors, including a small
population of delisted names to provide survivorship-bias awareness.
"""

from __future__ import annotations

import hashlib
import random
import string
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from qrt.utils.config import Config
from qrt.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sector / industry taxonomy
# ---------------------------------------------------------------------------

_SECTOR_INDUSTRIES: dict[str, list[str]] = {
    "Technology": [
        "Software—Application",
        "Software—Infrastructure",
        "Semiconductors",
        "Electronic Components",
        "IT Services & Consulting",
        "Internet Content & Information",
        "Hardware & Storage",
        "Communication Equipment",
    ],
    "Financials": [
        "Banks—Diversified",
        "Banks—Regional",
        "Asset Management",
        "Insurance—Diversified",
        "Capital Markets",
        "Financial Data & Technology",
        "Mortgage Finance",
    ],
    "Healthcare": [
        "Drug Manufacturers—General",
        "Biotechnology",
        "Medical Devices",
        "Managed Health Care",
        "Health Information Services",
        "Medical Diagnostics & Research",
    ],
    "Consumer": [
        "Internet Retail",
        "Specialty Retail",
        "Consumer Electronics",
        "Apparel Retail",
        "Restaurants",
        "Household Products",
        "Food Distribution",
        "Beverages—Non-Alcoholic",
    ],
    "Industrials": [
        "Aerospace & Defense",
        "Industrial Machinery",
        "Waste Management",
        "Railroads",
        "Trucking",
        "Construction & Engineering",
    ],
    "Energy": [
        "Oil & Gas E&P",
        "Oil & Gas Integrated",
        "Oil & Gas Midstream",
        "Oil & Gas Equipment & Services",
    ],
    "Materials": [
        "Specialty Chemicals",
        "Diversified Metals & Mining",
        "Gold",
        "Steel",
        "Agricultural Inputs",
    ],
}

# Realistic ticker prefixes per sector (first 1-4 chars used to seed names)
_TICKER_SEEDS: dict[str, list[str]] = {
    "Technology": [
        "ACME", "NOVA", "SYNT", "AXON", "PROX", "VECT", "ZYLO", "QUBT",
        "NXGN", "CLDX", "SFTW", "DVLP", "BYTE", "ALGN", "CSLT", "DSYS",
        "EDGE", "FLUX", "GIGA", "HLTH", "INFI", "JVLT", "KYPR", "LNKD",
        "MTRX", "NEXA", "OPCO", "PRGS", "QBTS", "RDNT", "SQSP", "TRNS",
        "ULVR", "VCTR", "WRKD", "XPND", "YEXT", "ZMBL", "ACTL", "BSYS",
    ],
    "Financials": [
        "FXBK", "ALPH", "PRMR", "CPTL", "MRKT", "BANC", "FNDC", "INVT",
        "CLRW", "DVRS", "EQUI", "FSTR", "GNRT", "HRTZ", "INDE", "JPCS",
        "KPTN", "LBTY", "MNCR", "NCMB", "OCVR", "PCFT", "QNTY", "RSVT",
        "STNR", "TRST",
    ],
    "Healthcare": [
        "BIOT", "HLTH", "PHMD", "MDVX", "GENX", "RXMD", "DIAG", "TBIO",
        "ABBT", "BDXT", "CEMD", "DPMD", "EXMD", "FXMD", "GXMD", "HXMD",
        "IXMD", "JXMD", "KXMD", "LXMD", "MXMD", "NXMD", "OXMD", "PXMD",
        "QXMD", "RXMA", "SXMD", "TXMD", "UXMD", "VXMD",
    ],
    "Consumer": [
        "RTLX", "GROC", "FASH", "RSTX", "BVRG", "HSHD", "CLTH", "DLVR",
        "AMZN", "SHPF", "CNSM", "DLUX", "ENTR", "FRTL", "GRND", "HRZN",
        "INSP", "JNTY", "KOOL", "LXRY", "MNDO", "NRTH", "OPTN", "PRME",
        "QHST", "RFRSH", "SMRT", "TDST",
    ],
    "Industrials": [
        "INDS", "MFGX", "AERO", "ENGG", "RALY", "CNST", "WSTE", "TRCK",
        "BLDG", "CIVL", "DFNS", "ENGX", "FCLG", "GLBL", "HVAC", "INTL",
        "JVLT", "KLMN", "LOGI", "MNFG", "NRGY", "OPRS",
    ],
    "Energy": [
        "ERGY", "OILX", "PIPE", "DRLG", "RFNG", "XPLO", "PETX", "NGAS",
        "CRDE", "DPRD", "ESTX", "FLNG", "GSPY", "HPET", "IPEX",
    ],
    "Materials": [
        "MTLS", "CHEM", "GOLD", "SLVR", "MTAL", "MING", "AGRI", "STLX",
        "BAUX", "CPPR", "DIAM", "ELMN", "FERR", "GRPH", "HLON",
    ],
}

_EXCHANGES = ["NYSE", "NASDAQ", "NYSE", "NYSE", "NASDAQ"]  # weighted

_COMPANY_SUFFIXES = [
    "Inc.", "Corp.", "Holdings Inc.", "Group Inc.",
    "Technologies Inc.", "Solutions Inc.", "Systems Corp.",
    "Enterprises Inc.", "International Corp.", "Ltd.",
]


def _make_cusip(seed: str) -> str:
    """Generate a deterministic but fake 9-char CUSIP-like string."""
    h = hashlib.md5(seed.encode()).hexdigest()
    return (h[:6] + h[6:8] + h[8]).upper()


def _make_isin(cusip: str, country: str = "US") -> str:
    return f"{country}{cusip}"


def _company_name(sector: str, ticker: str) -> str:
    sector_adj = {
        "Technology": ["Advanced", "Digital", "Intelligent", "Next-Gen", "Quantum", "Cloud"],
        "Financials": ["Premier", "Capital", "National", "First", "Global", "Alliance"],
        "Healthcare": ["BioMed", "GenPharma", "HealthCore", "MedTech", "ClinPath", "BioGen"],
        "Consumer": ["Prime", "Select", "National", "Fresh", "Urban", "Heritage"],
        "Industrials": ["Precision", "Global", "American", "Industrial", "National", "United"],
        "Energy": ["Petroleum", "Energy", "Resources", "Power", "Fuel", "Exploration"],
        "Materials": ["Advanced", "Core", "Specialty", "Prime", "Standard", "Integrated"],
    }
    adjs = sector_adj.get(sector, ["Global"])
    rng = random.Random(ticker)
    adj = rng.choice(adjs)
    suffix = rng.choice(_COMPANY_SUFFIXES)
    base = ticker.capitalize()
    return f"{adj} {base} {suffix}"


# ---------------------------------------------------------------------------
# Target security counts per sector
# ---------------------------------------------------------------------------

_SECTOR_TARGET_COUNTS: dict[str, int] = {
    "Technology": 35,
    "Financials": 25,
    "Healthcare": 25,
    "Consumer": 25,
    "Industrials": 22,
    "Energy": 10,
    "Materials": 13,
}
# Total active ≈ 155; we add ~15 delisted = 170 total

_DELISTED_EXTRA_PER_SECTOR: dict[str, int] = {
    "Technology": 4,
    "Financials": 3,
    "Healthcare": 3,
    "Consumer": 2,
    "Industrials": 2,
    "Energy": 1,
    "Materials": 1,
}


class SecurityMaster:
    """
    Maintains the reference data for all securities in the synthetic universe.

    Columns
    -------
    security_id    : int  — surrogate key
    ticker         : str
    cusip          : str
    isin           : str
    company_name   : str
    sector         : str
    industry       : str
    exchange       : str
    currency       : str  (always USD)
    country        : str  (always US)
    list_date      : date
    delist_date    : date | NaT
    is_active      : bool
    """

    COLUMNS = [
        "security_id", "ticker", "cusip", "isin", "company_name",
        "sector", "industry", "exchange", "currency", "country",
        "list_date", "delist_date", "is_active",
    ]

    def __init__(self, config: Optional[Config] = None, seed: int = 42) -> None:
        self._config = config or Config()
        self._seed = seed
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate and cache the security master DataFrame."""
        logger.info("Generating security master …")
        records = []
        security_id = 1

        start_date = pd.Timestamp(self._config.get("data.start_date", "2015-01-01")).date()
        end_date = pd.Timestamp(self._config.get("data.end_date", "2024-12-31")).date()

        rng = np.random.default_rng(self._seed)
        py_rng = random.Random(self._seed)

        for sector, count in _SECTOR_TARGET_COUNTS.items():
            industries = _SECTOR_INDUSTRIES[sector]
            ticker_pool = _TICKER_SEEDS[sector][:]

            # Generate active securities
            for i in range(count):
                ticker = ticker_pool[i % len(ticker_pool)]
                # Make tickers unique by appending a digit when recycling
                if i >= len(ticker_pool):
                    ticker = ticker[:3] + str(i - len(ticker_pool) + 1)

                # List date: between start_date and 2 years in
                list_offset = int(rng.integers(0, 365))
                list_date = start_date + timedelta(days=list_offset)

                industry = industries[i % len(industries)]
                exchange = py_rng.choice(_EXCHANGES)
                cusip = _make_cusip(f"{sector}{ticker}{i}")
                isin = _make_isin(cusip)

                records.append({
                    "security_id": security_id,
                    "ticker": ticker,
                    "cusip": cusip,
                    "isin": isin,
                    "company_name": _company_name(sector, ticker),
                    "sector": sector,
                    "industry": industry,
                    "exchange": exchange,
                    "currency": "USD",
                    "country": "US",
                    "list_date": list_date,
                    "delist_date": None,
                    "is_active": True,
                })
                security_id += 1

        # Generate delisted securities (survivorship bias population)
        for sector, dcount in _DELISTED_EXTRA_PER_SECTOR.items():
            industries = _SECTOR_INDUSTRIES[sector]
            ticker_pool = _TICKER_SEEDS[sector][:]

            for j in range(dcount):
                idx = 100 + j  # offset to avoid ticker collision
                ticker = "D" + ticker_pool[j % len(ticker_pool)][:3]
                if j > 0:
                    ticker = ticker[:4] + str(j)

                # Listed early in the sample period, delisted mid-way
                list_date = start_date + timedelta(days=int(rng.integers(0, 180)))
                delist_years = float(rng.uniform(1.5, 6.0))
                delist_date = list_date + timedelta(days=int(delist_years * 365))
                # Cap at end_date
                delist_date = min(delist_date, end_date - timedelta(days=30))

                industry = industries[j % len(industries)]
                exchange = py_rng.choice(_EXCHANGES)
                cusip = _make_cusip(f"DELIST{sector}{ticker}{j}")
                isin = _make_isin(cusip)

                records.append({
                    "security_id": security_id,
                    "ticker": ticker,
                    "cusip": cusip,
                    "isin": isin,
                    "company_name": _company_name(sector, f"DL{ticker}"),
                    "sector": sector,
                    "industry": industry,
                    "exchange": exchange,
                    "currency": "USD",
                    "country": "US",
                    "list_date": list_date,
                    "delist_date": delist_date,
                    "is_active": False,
                })
                security_id += 1

        df = pd.DataFrame(records, columns=self.COLUMNS)
        df["list_date"] = pd.to_datetime(df["list_date"])
        df["delist_date"] = pd.to_datetime(df["delist_date"])
        df = df.set_index("security_id", drop=False)

        self._df = df
        logger.info(
            "Security master: %d total (%d active, %d delisted)",
            len(df),
            df["is_active"].sum(),
            (~df["is_active"]).sum(),
        )
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.generate()
        return self._df  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Point-in-time helpers
    # ------------------------------------------------------------------

    def as_of(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        Return securities that were *listed* (and not yet delisted) on
        ``as_of_date``.  This is the canonical point-in-time slice.
        """
        df = self.df
        listed = df["list_date"] <= as_of_date
        still_active = df["delist_date"].isna() | (df["delist_date"] > as_of_date)
        return df[listed & still_active].copy()

    def active_ids(self, as_of_date: pd.Timestamp) -> list[int]:
        """Return list of security_ids active on ``as_of_date``."""
        return self.as_of(as_of_date)["security_id"].tolist()

    def get_sector(self, security_id: int) -> str:
        return self.df.loc[security_id, "sector"]

    def get_by_ticker(self, ticker: str) -> pd.Series:
        matches = self.df[self.df["ticker"] == ticker]
        if matches.empty:
            raise KeyError(f"Ticker {ticker!r} not found in security master")
        return matches.iloc[0]

    def sector_map(self) -> dict[int, str]:
        """Return {security_id: sector} mapping."""
        return self.df.set_index("security_id")["sector"].to_dict()
