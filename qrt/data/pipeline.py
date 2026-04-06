"""
DataPipeline — orchestrates generation and persistence of all data artifacts.

Workflow
--------
1. Generate Security Master
2. Generate Market Data (OHLCV daily bars)
3. Calculate Returns
4. Build Universes (100, 150, 200, 300)
5. Persist everything to Parquet files under data/parquet/
6. load_dataset() reloads from Parquet, avoiding redundant computation.

Directory layout under <project_root>/data/parquet/:
  security_master.parquet
  market_data.parquet
  returns.parquet
  universe_100.parquet
  universe_150.parquet
  universe_200.parquet
  universe_300.parquet
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from qrt.utils.config import Config
from qrt.utils.logger import get_logger
from qrt.data.security_master import SecurityMaster
from qrt.data.market_data import MarketDataGenerator
from qrt.data.returns import ReturnsCalculator
from qrt.data.universe import UniverseConstructor
from qrt.data.real_data import RealDataFetcher

logger = get_logger(__name__)

# Names of parquet files (relative to parquet_dir)
_PARQUET_FILES = {
    "security_master": "security_master.parquet",
    "market_data": "market_data.parquet",
    "returns": "returns.parquet",
    "universe_100": "universe_100.parquet",
    "universe_150": "universe_150.parquet",
    "universe_200": "universe_200.parquet",
    "universe_300": "universe_300.parquet",
}


class DataPipeline:
    """
    End-to-end data pipeline for the quantitative research platform.

    Parameters
    ----------
    config : Config, optional
        If not provided, loads from the default config path.
    seed : int
        Master random seed; sub-seeds are derived deterministically.
    project_root : str | Path, optional
        Root of the project.  Parquet files are written under
        <project_root>/<config.data.parquet_dir>.
        Defaults to the repo root (three levels above this file).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        seed: int = 42,
        project_root: Optional[str | Path] = None,
        source: str = "synthetic",
    ) -> None:
        self._config = config or Config()
        self._seed = seed
        self._source = source  # "synthetic" or "real"

        if project_root is None:
            # Resolve to <repo_root>
            project_root = Path(__file__).parent.parent.parent

        self._project_root = Path(project_root)
        parquet_rel = self._config.get("data.parquet_dir", "data/parquet")
        self._parquet_dir = self._project_root / parquet_rel

        # Lazily populated
        self._security_master: Optional[pd.DataFrame] = None
        self._market_data: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None
        self._universes: Optional[dict[int, pd.DataFrame]] = None

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(self, force_regenerate: bool = False) -> "DataPipeline":
        """
        Execute the full pipeline.

        If Parquet files already exist and ``force_regenerate`` is False,
        the pipeline loads from disk instead of regenerating.

        Parameters
        ----------
        force_regenerate : bool
            When True, regenerate all data even if Parquet files exist.

        Returns
        -------
        self  (fluent interface)
        """
        if not force_regenerate and self._all_parquet_exist():
            logger.info("Parquet files found — loading from disk (use force_regenerate=True to rebuild).")
            self.load_dataset()
            return self

        logger.info("=" * 60)
        logger.info("Starting DataPipeline (source=%s, seed=%d)", self._source, self._seed)
        logger.info("=" * 60)
        t0 = time.perf_counter()

        if self._source == "real":
            # Download real market data from Yahoo Finance
            self._security_master, self._market_data = self._step_real_data()
        else:
            # Step 1 — Security Master (synthetic)
            self._security_master = self._step_security_master()
            # Step 2 — Market Data (synthetic)
            self._market_data = self._step_market_data()

        # Step 3 — Returns (same logic for both sources)
        self._returns = self._step_returns()

        # Step 4 — Universes (same logic for both sources)
        self._universes = self._step_universes()

        # Step 5 — Persist
        self._persist()

        elapsed = time.perf_counter() - t0
        logger.info("Pipeline complete in %.1f s", elapsed)
        return self

    def load_dataset(self) -> "DataPipeline":
        """
        Load all datasets from Parquet.

        Raises
        ------
        FileNotFoundError
            If any required Parquet file is missing.
        """
        logger.info("Loading dataset from %s", self._parquet_dir)
        missing = [
            name
            for name, fname in _PARQUET_FILES.items()
            if not (self._parquet_dir / fname).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing Parquet files: {missing}. Run pipeline.run() first."
            )

        self._security_master = pd.read_parquet(self._parquet_dir / _PARQUET_FILES["security_master"])
        self._market_data = pd.read_parquet(self._parquet_dir / _PARQUET_FILES["market_data"])
        self._returns = pd.read_parquet(self._parquet_dir / _PARQUET_FILES["returns"])

        self._universes = {}
        for size in UniverseConstructor.SUPPORTED_SIZES:
            key = f"universe_{size}"
            self._universes[size] = pd.read_parquet(self._parquet_dir / _PARQUET_FILES[key])

        logger.info(
            "Loaded: %d securities | %d market rows | %d return rows",
            len(self._security_master),
            len(self._market_data),
            len(self._returns),
        )
        return self

    # ------------------------------------------------------------------
    # Property accessors
    # ------------------------------------------------------------------

    @property
    def security_master(self) -> pd.DataFrame:
        self._require("_security_master", "security_master")
        return self._security_master  # type: ignore[return-value]

    @property
    def market_data(self) -> pd.DataFrame:
        self._require("_market_data", "market_data")
        return self._market_data  # type: ignore[return-value]

    @property
    def returns(self) -> pd.DataFrame:
        self._require("_returns", "returns")
        return self._returns  # type: ignore[return-value]

    @property
    def universes(self) -> dict[int, pd.DataFrame]:
        self._require("_universes", "universes")
        return self._universes  # type: ignore[return-value]

    def get_universe(self, size: int, date: pd.Timestamp) -> list[int]:
        """
        Return security_ids in universe<size> on ``date`` (most recent
        rebalance ≤ date).
        """
        uni_df = self.universes[size]
        mask = uni_df["rebalance_date"] <= date
        if not mask.any():
            return []
        latest_rb = uni_df.loc[mask, "rebalance_date"].max()
        return uni_df.loc[uni_df["rebalance_date"] == latest_rb, "security_id"].tolist()

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _step_real_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Step 1-2/4 — Fetching Real Market Data (Yahoo Finance)")
        start = self._config.get("data.real_start_date", "2010-01-01")
        end = self._config.get("data.real_end_date", None)
        fetcher = RealDataFetcher(config=self._config, start_date=start, end_date=end)
        return fetcher.fetch()

    def _step_security_master(self) -> pd.DataFrame:
        logger.info("Step 1/4 — Security Master")
        sm = SecurityMaster(config=self._config, seed=self._seed)
        return sm.generate()

    def _step_market_data(self) -> pd.DataFrame:
        logger.info("Step 2/4 — Market Data")
        assert self._security_master is not None
        gen = MarketDataGenerator(
            security_master=self._security_master,
            config=self._config,
            seed=self._seed + 1,
        )
        return gen.generate()

    def _step_returns(self) -> pd.DataFrame:
        logger.info("Step 3/4 — Returns")
        assert self._market_data is not None and self._security_master is not None
        calc = ReturnsCalculator(
            market_data=self._market_data,
            security_master=self._security_master,
            config=self._config,
        )
        return calc.calculate()

    def _step_universes(self) -> dict[int, pd.DataFrame]:
        logger.info("Step 4/4 — Universes")
        assert self._security_master is not None and self._market_data is not None
        uc = UniverseConstructor(
            security_master=self._security_master,
            market_data=self._market_data,
            config=self._config,
        )
        uc.build()
        return {size: uc.as_dataframe(size) for size in UniverseConstructor.SUPPORTED_SIZES}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Writing Parquet files to %s", self._parquet_dir)

        self._write_parquet(self._security_master, "security_master")
        self._write_parquet(self._market_data, "market_data")
        self._write_parquet(self._returns, "returns")

        for size, df in self._universes.items():  # type: ignore[union-attr]
            self._write_parquet(df, f"universe_{size}")

        logger.info("All Parquet files written.")

    def _write_parquet(self, df: pd.DataFrame, key: str) -> None:
        path = self._parquet_dir / _PARQUET_FILES[key]
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        size_mb = path.stat().st_size / 1e6
        logger.info("  Wrote %-30s  (%.1f MB, %d rows)", path.name, size_mb, len(df))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_parquet_exist(self) -> bool:
        return all(
            (self._parquet_dir / fname).exists()
            for fname in _PARQUET_FILES.values()
        )

    def _require(self, attr: str, name: str) -> None:
        if getattr(self, attr) is None:
            raise RuntimeError(
                f"'{name}' is not loaded. Call pipeline.run() or pipeline.load_dataset() first."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a dict of summary statistics for quick sanity checks."""
        out: dict[str, Any] = {}

        if self._security_master is not None:
            sm = self._security_master
            out["n_securities_total"] = len(sm)
            out["n_securities_active"] = int(sm["is_active"].sum())
            out["n_securities_delisted"] = int((~sm["is_active"]).sum())
            out["sectors"] = sm["sector"].value_counts().to_dict()

        if self._market_data is not None:
            md = self._market_data
            out["market_data_rows"] = len(md)
            out["market_data_date_range"] = (str(md["date"].min()), str(md["date"].max()))

        if self._returns is not None:
            ret = self._returns
            out["returns_rows"] = len(ret)
            median_adj_ret = float(ret["ret_adj"].dropna().median())
            out["median_daily_ret_adj"] = round(median_adj_ret, 6)

        if self._universes is not None:
            out["universe_sizes"] = {}
            for size, df in self._universes.items():
                n_dates = df["rebalance_date"].nunique()
                avg_members = df.groupby("rebalance_date")["security_id"].count().mean()
                out["universe_sizes"][size] = {
                    "rebalance_dates": n_dates,
                    "avg_members": round(float(avg_members), 1),
                }

        return out
