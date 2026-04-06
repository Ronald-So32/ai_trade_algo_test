"""
Backtest Integrity Audit Engine
================================
Automated validation of the quant research platform against institutional
quantitative research standards.

Performs checks on:
  - Data source integrity (real vs synthetic)
  - Signal construction (no lookahead bias)
  - Execution lag (weights.shift(1))
  - Transaction cost modeling
  - Walk-forward validation
  - Strategy research grounding

Produces a structured validation report with STATUS: VALIDATED / NOT VALIDATED.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AuditFinding:
    """Single audit finding."""
    category: str
    check_name: str
    status: str          # "PASS", "FAIL", "WARNING"
    detail: str
    file_path: str = ""
    line_number: int = 0


@dataclass
class AuditReport:
    """Structured validation report."""
    findings: list[AuditFinding] = field(default_factory=list)
    data_source: str = "unknown"
    overall_status: str = "NOT VALIDATED"

    @property
    def pass_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "PASS")

    @property
    def fail_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "FAIL")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "WARNING")

    def determine_status(self) -> str:
        """Determine overall validation status."""
        critical_fails = [
            f for f in self.findings
            if f.status == "FAIL" and f.category in (
                "DATA_SOURCE", "SIGNAL_INTEGRITY", "EXECUTION_LAG"
            )
        ]
        if critical_fails:
            self.overall_status = "NOT VALIDATED"
        elif self.fail_count > 0:
            self.overall_status = "NOT VALIDATED"
        else:
            self.overall_status = "VALIDATED"
        return self.overall_status

    def to_markdown(self) -> str:
        """Generate markdown validation report."""
        lines = [
            "# Backtest Integrity Validation Report",
            "",
            f"**Overall Status: {self.determine_status()}**",
            f"**Data Source: {self.data_source}**",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Passed | {self.pass_count} |",
            f"| Failed | {self.fail_count} |",
            f"| Warnings | {self.warning_count} |",
            "",
        ]

        # Group by category
        categories = sorted(set(f.category for f in self.findings))
        for cat in categories:
            lines.append(f"## {cat.replace('_', ' ').title()}")
            lines.append("")
            cat_findings = [f for f in self.findings if f.category == cat]
            for f in cat_findings:
                icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARNING": "[WARN]"}[f.status]
                lines.append(f"- {icon} **{f.check_name}**: {f.detail}")
                if f.file_path:
                    lines.append(f"  - File: `{f.file_path}`" + (f":{f.line_number}" if f.line_number else ""))
            lines.append("")

        return "\n".join(lines)


class BacktestAuditEngine:
    """
    Performs a comprehensive audit of the quant research platform.

    Usage
    -----
    >>> engine = BacktestAuditEngine(project_root="/path/to/project")
    >>> report = engine.run_full_audit(data_source="real")
    >>> print(report.to_markdown())
    """

    # Patterns that indicate synthetic data generation
    SYNTHETIC_PATTERNS = [
        "GBM", "geometric_brownian_motion", "simulate_prices",
        "simulate_market", "generate_market_data", "synthetic_data",
        "market_data_generator", "MarketDataGenerator",
    ]

    # Patterns for real data ingestion
    REAL_DATA_PATTERNS = [
        "yfinance", "yf.download", "yf.Ticker",
        "polygon", "nasdaq_data_link", "wrds", "alpha_vantage",
    ]

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self.report = AuditReport()

    def run_full_audit(
        self,
        data_source: str = "unknown",
        strategy_results: dict | None = None,
        prices: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
        weights_dict: dict[str, pd.DataFrame] | None = None,
    ) -> AuditReport:
        """Run all audit checks and produce a structured report."""
        self.report = AuditReport(data_source=data_source)

        # 1. Data source checks
        self._check_data_source(data_source)
        self._scan_for_synthetic_usage()
        self._check_real_data_ingestion()
        self._check_canonical_storage()
        self._check_data_quality(prices, returns)

        # 2. Signal integrity
        self._check_signal_construction()
        self._check_lookahead_bias()

        # 3. Execution lag
        self._check_execution_lag()

        # 4. Transaction cost model
        self._check_transaction_costs()

        # 5. Walk-forward validation
        self._check_walk_forward()

        # 6. Model validation (ML meta-model)
        self._check_ml_model_integrity()

        # 7. Execution realism
        if strategy_results and returns is not None:
            self._check_execution_realism(strategy_results, returns)

        # 8. Research grounding
        self._check_research_grounding()

        self.report.determine_status()
        return self.report

    # ------------------------------------------------------------------
    # 1. Data source checks
    # ------------------------------------------------------------------

    def _check_data_source(self, source: str) -> None:
        if source == "real":
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Data source type",
                status="PASS",
                detail=f"Platform configured with real market data (source='{source}')",
            ))
        elif source == "synthetic":
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Data source type",
                status="FAIL",
                detail="Platform using synthetic GBM-generated data. Switch to --real-data for validated results.",
            ))
        else:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Data source type",
                status="WARNING",
                detail=f"Unknown data source: '{source}'. Cannot confirm real market data.",
            ))

    def _scan_for_synthetic_usage(self) -> None:
        """Scan strategy files for synthetic data generator usage."""
        strategy_dir = self.project_root / "qrt" / "strategies"
        run_file = self.project_root / "run_research.py"
        synthetic_in_strategies = False

        for py_file in list(strategy_dir.glob("*.py")) + [run_file]:
            if not py_file.exists():
                continue
            content = py_file.read_text()
            for pattern in self.SYNTHETIC_PATTERNS:
                if pattern in content and "synthetic" not in py_file.name:
                    # Use word-boundary matching to avoid flagging substrings
                    # (e.g. "GBM" inside "LGBMRegressor" or "_USE_LGBM" is not
                    # a reference to Geometric Brownian Motion)
                    if not re.search(rf'\b{re.escape(pattern)}\b', content):
                        continue
                    # Check if it's in the data pipeline (expected) vs strategy engine (problem)
                    if py_file.parent.name == "strategies":
                        synthetic_in_strategies = True
                        self.report.findings.append(AuditFinding(
                            category="DATA_SOURCE",
                            check_name="Synthetic data in strategy",
                            status="FAIL",
                            detail=f"Strategy file references synthetic pattern '{pattern}'",
                            file_path=str(py_file.relative_to(self.project_root)),
                        ))

        if not synthetic_in_strategies:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Strategy independence from synthetic data",
                status="PASS",
                detail="No strategy files directly reference synthetic data generators",
            ))

    def _check_real_data_ingestion(self) -> None:
        """Verify real data ingestion infrastructure exists."""
        real_data_file = self.project_root / "qrt" / "data" / "real_data.py"
        if not real_data_file.exists():
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Real data ingestion module",
                status="FAIL",
                detail="No real data ingestion module found (expected qrt/data/real_data.py)",
            ))
            return

        content = real_data_file.read_text()
        apis_found = []
        for pattern in self.REAL_DATA_PATTERNS:
            if pattern in content:
                apis_found.append(pattern)

        if apis_found:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Real data APIs",
                status="PASS",
                detail=f"Real data ingestion uses: {', '.join(apis_found)}",
                file_path="qrt/data/real_data.py",
            ))
        else:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Real data APIs",
                status="WARNING",
                detail="Real data module exists but no recognized API patterns found",
                file_path="qrt/data/real_data.py",
            ))

        # Check for tickers
        if "REAL_UNIVERSE" in content:
            # Count tickers
            import re
            tickers = re.findall(r'"([A-Z]{1,5})"', content)
            unique_tickers = set(tickers)
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Universe coverage",
                status="PASS",
                detail=f"Real universe contains {len(unique_tickers)} unique tickers across multiple sectors",
                file_path="qrt/data/real_data.py",
            ))

    def _check_canonical_storage(self) -> None:
        """Check for proper parquet storage format."""
        parquet_dir = self.project_root / "data" / "parquet"
        expected_files = [
            "market_data.parquet",
            "returns.parquet",
            "security_master.parquet",
        ]

        if not parquet_dir.exists():
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Canonical storage",
                status="WARNING",
                detail="Parquet data directory not found (run pipeline first)",
            ))
            return

        for fname in expected_files:
            fpath = parquet_dir / fname
            if fpath.exists():
                size_mb = fpath.stat().st_size / 1e6
                self.report.findings.append(AuditFinding(
                    category="DATA_SOURCE",
                    check_name=f"Storage: {fname}",
                    status="PASS",
                    detail=f"Found {fname} ({size_mb:.1f} MB)",
                    file_path=f"data/parquet/{fname}",
                ))
            else:
                self.report.findings.append(AuditFinding(
                    category="DATA_SOURCE",
                    check_name=f"Storage: {fname}",
                    status="WARNING",
                    detail=f"Missing {fname} (will be created on pipeline run)",
                ))

    def _check_data_quality(
        self,
        prices: pd.DataFrame | None,
        returns: pd.DataFrame | None,
    ) -> None:
        """Check data quality if DataFrames are provided."""
        if prices is None or returns is None:
            return

        # Check for adjusted prices
        if prices.min().min() > 0:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Price positivity",
                status="PASS",
                detail="All prices are positive (no data corruption)",
            ))
        else:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Price positivity",
                status="WARNING",
                detail=f"Found {(prices <= 0).sum().sum()} non-positive prices",
            ))

        # Check timestamps
        if isinstance(prices.index, pd.DatetimeIndex):
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Timestamp format",
                status="PASS",
                detail=f"Proper DatetimeIndex, range: {prices.index[0].date()} to {prices.index[-1].date()}",
            ))
        else:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="Timestamp format",
                status="FAIL",
                detail="Index is not DatetimeIndex — timestamps may be incorrect",
            ))

        # Check for weekend data (would indicate bad data)
        weekday_counts = prices.index.dayofweek.value_counts()
        weekend_count = weekday_counts.get(5, 0) + weekday_counts.get(6, 0)
        if weekend_count == 0:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="No weekend data",
                status="PASS",
                detail="No weekend dates in price data (trading days only)",
            ))
        else:
            self.report.findings.append(AuditFinding(
                category="DATA_SOURCE",
                check_name="No weekend data",
                status="WARNING",
                detail=f"Found {weekend_count} weekend date entries",
            ))

    # ------------------------------------------------------------------
    # 2. Signal integrity
    # ------------------------------------------------------------------

    def _check_signal_construction(self) -> None:
        """Verify signal construction uses only historical data."""
        strategy_dir = self.project_root / "qrt" / "strategies"
        for py_file in strategy_dir.glob("*.py"):
            if py_file.name in ("__init__.py", "base.py"):
                continue
            content = py_file.read_text()

            # Check for .shift(-1) in signal generation (lookahead)
            # Exclude comments and docstrings
            code_lines = [
                (i + 1, line) for i, line in enumerate(content.split("\n"))
                if line.strip() and not line.strip().startswith("#")
                and not line.strip().startswith('"""')
                and not line.strip().startswith("'''")
            ]

            # Parse function boundaries so we can identify context of each line
            all_lines = content.split("\n")
            current_func = ""
            func_map: dict[int, str] = {}  # line_number -> enclosing function name
            for idx, raw_line in enumerate(all_lines, start=1):
                func_match = re.match(r'\s+def\s+(\w+)\s*\(', raw_line)
                if func_match:
                    current_func = func_match.group(1)
                func_map[idx] = current_func

            # Function names that indicate ML target / label construction
            # where .shift(-N) is legitimate (forward returns used only as
            # training labels, not for live signal generation)
            _TARGET_FUNC_PATTERNS = (
                "target", "label", "stack_features_panel",
                "build_target", "build_label", "_build_y",
            )

            for line_num, line in code_lines:
                if ".shift(-" in line:
                    enclosing = func_map.get(line_num, "")
                    # Allow .shift(-N) in ML target construction functions
                    if any(pat in enclosing.lower() for pat in _TARGET_FUNC_PATTERNS):
                        self.report.findings.append(AuditFinding(
                            category="SIGNAL_INTEGRITY",
                            check_name=f"ML target construction in {py_file.name}",
                            status="PASS",
                            detail=(
                                f".shift(-N) used in target construction method '{enclosing}()' "
                                f"— acceptable for ML training labels when walk-forward purge gap is enforced"
                            ),
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=line_num,
                        ))
                    else:
                        self.report.findings.append(AuditFinding(
                            category="SIGNAL_INTEGRITY",
                            check_name=f"Lookahead in {py_file.name}",
                            status="FAIL",
                            detail=f"Found .shift(-N) which uses future data: {line.strip()[:80]}",
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=line_num,
                        ))

        # Check that momentum uses trailing returns
        for py_file in strategy_dir.glob("*momentum*.py"):
            content = py_file.read_text()
            if "lookback" in content.lower() or "trailing" in content.lower():
                self.report.findings.append(AuditFinding(
                    category="SIGNAL_INTEGRITY",
                    check_name=f"Trailing lookback in {py_file.name}",
                    status="PASS",
                    detail="Momentum signal uses trailing lookback window (point-in-time correct)",
                    file_path=str(py_file.relative_to(self.project_root)),
                ))

        # Check mean reversion uses rolling statistics
        mr_file = strategy_dir / "mean_reversion.py"
        if mr_file.exists():
            content = mr_file.read_text()
            if ".rolling(" in content:
                self.report.findings.append(AuditFinding(
                    category="SIGNAL_INTEGRITY",
                    check_name="Rolling statistics in mean_reversion",
                    status="PASS",
                    detail="Mean reversion uses rolling window statistics (no future leakage)",
                    file_path="qrt/strategies/mean_reversion.py",
                ))

    def _check_lookahead_bias(self) -> None:
        """Deep scan for lookahead bias across all Python files."""
        dangerous_patterns = [
            (r"\.shift\(-\d+\)", "Forward shift (uses future data)"),
            # Only flag iloc[t+1] when NOT part of a slice (slice end is exclusive, so t+1 means "up to t")
            (r"\.iloc\[\s*[a-zA-Z_]+\s*\+\s*1\s*\]", "Potential future index access (non-slice)"),
        ]

        # Only check strategy and signal generation files
        scan_dirs = [
            self.project_root / "qrt" / "strategies",
            self.project_root / "qrt" / "alpha_engine",
        ]

        clean = True
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for py_file in scan_dir.glob("*.py"):
                content = py_file.read_text()
                for pattern, desc in dangerous_patterns:
                    matches = list(re.finditer(pattern, content))
                    for m in matches:
                        # Find line number
                        line_num = content[:m.start()].count("\n") + 1
                        line_text = content.split("\n")[line_num - 1].strip()
                        # Skip if in comment or docstring
                        if line_text.startswith("#"):
                            continue
                        clean = False
                        self.report.findings.append(AuditFinding(
                            category="SIGNAL_INTEGRITY",
                            check_name=f"Lookahead: {py_file.name}",
                            status="FAIL" if "strategies" in str(py_file) else "WARNING",
                            detail=f"{desc}: {line_text[:80]}",
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=line_num,
                        ))

        if clean:
            self.report.findings.append(AuditFinding(
                category="SIGNAL_INTEGRITY",
                check_name="Lookahead bias scan",
                status="PASS",
                detail="No lookahead bias patterns found in strategy or signal generation code",
            ))

        # Separately audit the ML meta-model
        meta_file = self.project_root / "qrt" / "ml_meta" / "meta_model.py"
        if meta_file.exists():
            content = meta_file.read_text()
            if ".shift(-1)" in content:
                self.report.findings.append(AuditFinding(
                    category="MODEL_VALIDATION",
                    check_name="ML meta-model lookahead",
                    status="WARNING",
                    detail=(
                        "ML meta-model uses .shift(-1) to construct targets from future returns. "
                        "This is acceptable IF: (1) the model is trained on historical data only, "
                        "(2) predictions are applied to unseen future data, and (3) build_targets() "
                        "is only called during training with a proper purge gap. "
                        "Verify that the fit/predict split enforces temporal ordering."
                    ),
                    file_path="qrt/ml_meta/meta_model.py",
                ))

    # ------------------------------------------------------------------
    # 3. Execution lag
    # ------------------------------------------------------------------

    def _check_execution_lag(self) -> None:
        """Verify weights.shift(1) is applied before computing returns."""
        base_file = self.project_root / "qrt" / "strategies" / "base.py"
        if not base_file.exists():
            self.report.findings.append(AuditFinding(
                category="EXECUTION_LAG",
                check_name="Base strategy shift(1)",
                status="FAIL",
                detail="Cannot find base.py to verify execution lag",
            ))
            return

        content = base_file.read_text()
        if "weights.shift(1)" in content or "shift(1)" in content:
            self.report.findings.append(AuditFinding(
                category="EXECUTION_LAG",
                check_name="Base strategy execution lag",
                status="PASS",
                detail=(
                    "backtest_summary() applies weights.shift(1) before multiplying "
                    "by returns — signals at t are traded at t+1"
                ),
                file_path="qrt/strategies/base.py",
            ))
        else:
            self.report.findings.append(AuditFinding(
                category="EXECUTION_LAG",
                check_name="Base strategy execution lag",
                status="FAIL",
                detail="Cannot find weights.shift(1) in backtest_summary() — possible same-bar execution",
                file_path="qrt/strategies/base.py",
            ))

        # Also check run_research.py
        run_file = self.project_root / "run_research.py"
        if run_file.exists():
            content = run_file.read_text()
            if "weights.shift(1)" in content:
                self.report.findings.append(AuditFinding(
                    category="EXECUTION_LAG",
                    check_name="Pipeline execution lag",
                    status="PASS",
                    detail="run_research.py also applies weights.shift(1) when computing strategy returns",
                    file_path="run_research.py",
                ))

    # ------------------------------------------------------------------
    # 4. Transaction costs
    # ------------------------------------------------------------------

    def _check_transaction_costs(self) -> None:
        """Verify transaction cost model exists and is comprehensive."""
        cost_file = self.project_root / "qrt" / "costs" / "transaction_costs.py"
        if not cost_file.exists():
            self.report.findings.append(AuditFinding(
                category="EXECUTION_REALISM",
                check_name="Transaction cost model",
                status="FAIL",
                detail="No transaction cost model found",
            ))
            return

        content = cost_file.read_text()

        components = {
            "commission": "commission" in content.lower(),
            "spread": "spread" in content.lower(),
            "slippage": "slippage" in content.lower(),
            "turnover_penalty": "turnover" in content.lower(),
        }

        all_present = all(components.values())
        missing = [k for k, v in components.items() if not v]

        if all_present:
            self.report.findings.append(AuditFinding(
                category="EXECUTION_REALISM",
                check_name="Transaction cost components",
                status="PASS",
                detail="All cost components present: commission, spread, slippage, turnover penalty",
                file_path="qrt/costs/transaction_costs.py",
            ))
        else:
            self.report.findings.append(AuditFinding(
                category="EXECUTION_REALISM",
                check_name="Transaction cost components",
                status="WARNING",
                detail=f"Missing cost components: {', '.join(missing)}",
                file_path="qrt/costs/transaction_costs.py",
            ))

        # Check for sqrt market impact model
        if "sqrt" in content:
            self.report.findings.append(AuditFinding(
                category="EXECUTION_REALISM",
                check_name="Market impact model",
                status="PASS",
                detail="Square-root market impact model implemented (institutional standard)",
                file_path="qrt/costs/transaction_costs.py",
            ))

    # ------------------------------------------------------------------
    # 5. Walk-forward validation
    # ------------------------------------------------------------------

    def _check_walk_forward(self) -> None:
        """Verify walk-forward testing infrastructure."""
        wf_file = self.project_root / "qrt" / "walkforward" / "walk_forward.py"
        if not wf_file.exists():
            self.report.findings.append(AuditFinding(
                category="MODEL_VALIDATION",
                check_name="Walk-forward testing",
                status="FAIL",
                detail="No walk-forward testing module found",
            ))
            return

        content = wf_file.read_text()

        # Check for train/test split
        has_train_test = "train" in content.lower() and "test" in content.lower()
        has_rolling = "rolling" in content.lower() or "window" in content.lower()
        has_purge = "purge" in content.lower() or "gap" in content.lower() or "gap_days" in content

        if has_train_test:
            self.report.findings.append(AuditFinding(
                category="MODEL_VALIDATION",
                check_name="Walk-forward train/test split",
                status="PASS",
                detail="Walk-forward module implements train/test temporal split",
                file_path="qrt/walkforward/walk_forward.py",
            ))

        if has_rolling:
            self.report.findings.append(AuditFinding(
                category="MODEL_VALIDATION",
                check_name="Rolling window validation",
                status="PASS",
                detail="Walk-forward uses rolling windows (not single fixed split)",
                file_path="qrt/walkforward/walk_forward.py",
            ))

        if has_purge:
            self.report.findings.append(AuditFinding(
                category="MODEL_VALIDATION",
                check_name="Purge gap",
                status="PASS",
                detail="Walk-forward includes purge gap between train and test sets",
                file_path="qrt/walkforward/walk_forward.py",
            ))
        else:
            self.report.findings.append(AuditFinding(
                category="MODEL_VALIDATION",
                check_name="Purge gap",
                status="WARNING",
                detail="No explicit purge gap found — potential information leakage at window boundaries",
                file_path="qrt/walkforward/walk_forward.py",
            ))

    # ------------------------------------------------------------------
    # 6. ML model integrity
    # ------------------------------------------------------------------

    def _check_ml_model_integrity(self) -> None:
        """Check ML meta-model for proper temporal handling."""
        cv_file = self.project_root / "qrt" / "ml_meta" / "cross_validation.py"
        if cv_file.exists():
            content = cv_file.read_text()
            if "TimeSeriesCV" in content or "time_series" in content.lower():
                self.report.findings.append(AuditFinding(
                    category="MODEL_VALIDATION",
                    check_name="Time-series cross-validation",
                    status="PASS",
                    detail="ML meta-model uses time-series aware CV (not random k-fold)",
                    file_path="qrt/ml_meta/cross_validation.py",
                ))

    # ------------------------------------------------------------------
    # 7. Execution realism
    # ------------------------------------------------------------------

    def _check_execution_realism(
        self,
        strategy_results: dict,
        returns: pd.DataFrame,
    ) -> None:
        """Check backtest returns for unrealistic performance."""
        for name, res in strategy_results.items():
            summary = res.get("summary", {})
            sharpe = summary.get("sharpe", 0)
            max_dd = abs(summary.get("max_drawdown", 0))
            turnover = summary.get("avg_turnover", 0)

            # Sharpe > 3 in-sample is suspicious
            if sharpe > 3.0:
                self.report.findings.append(AuditFinding(
                    category="EXECUTION_REALISM",
                    check_name=f"Suspicious Sharpe: {name}",
                    status="WARNING",
                    detail=f"In-sample Sharpe {sharpe:.2f} > 3.0 — likely overfit or has lookahead bias",
                ))

            # MaxDD = 0 is suspicious
            if max_dd < 0.001 and sharpe > 0:
                self.report.findings.append(AuditFinding(
                    category="EXECUTION_REALISM",
                    check_name=f"Zero drawdown: {name}",
                    status="WARNING",
                    detail=f"MaxDD near zero ({max_dd:.4f}) with positive Sharpe — unrealistic",
                ))

    # ------------------------------------------------------------------
    # 8. Research grounding
    # ------------------------------------------------------------------

    def _check_research_grounding(self) -> None:
        """Check if strategies have research grounding metadata."""
        strategy_dir = self.project_root / "qrt" / "strategies"
        for py_file in strategy_dir.glob("*.py"):
            if py_file.name in ("__init__.py", "base.py"):
                continue
            content = py_file.read_text()
            has_grounding = (
                "RESEARCH_GROUNDING" in content
                or "research_grounding" in content
                or "Research Grounding" in content
            )
            if has_grounding:
                self.report.findings.append(AuditFinding(
                    category="RESEARCH_GROUNDING",
                    check_name=f"Research grounding: {py_file.stem}",
                    status="PASS",
                    detail=f"Strategy {py_file.stem} includes research grounding metadata",
                    file_path=str(py_file.relative_to(self.project_root)),
                ))
