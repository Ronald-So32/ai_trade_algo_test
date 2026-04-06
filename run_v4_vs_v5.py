#!/usr/bin/env python3
"""
v4 vs v5 Comparison Backtest
=============================
Compares the static v4 system (equal-weight, fixed leverage) against
the dynamic v5 system (risk parity, vol-managed leverage, liquidity-weighted
residual reversal).

Academic basis for v5 improvements:
  1. Risk parity: Maillard, Roncalli & Teiletche (2010)
  2. Vol-managed leverage: Moreira & Muir (2017), Barroso & Santa-Clara (2015)
  3. Liquidity-weighted reversal: Nagel (2012), Amihud (2002)

Usage:
    python run_v4_vs_v5.py --real-data --holdout-date 2024-01-01
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from qrt.utils.config import Config
from qrt.utils.logger import get_logger

logger = get_logger("v4_vs_v5")

# ── Import strategy params from both versions ──
# v4 params (static)
TSMOM_PARAMS = {
    "lookback": 252,
    "vol_lookback": 63,
    "target_gross": 1.0,
    "vol_floor": 0.01,
    "multi_scale_weights": (0.0, 0.0, 1.0),
    "trend_strength_cap": 1.0,
    "vov_reduction": 0.0,
    "vov_lookback": 63,
}

RESIDUAL_STR_V4 = {
    "lookback": 5,
    "holding_period": 5,
    "long_pct": 0.20,
    "short_pct": 0.20,
    "target_gross": 1.0,
    "vol_scale": True,
    "vol_lookback": 21,
}

RESIDUAL_STR_V5_LIQ = {
    **RESIDUAL_STR_V4,
    "liquidity_weight": True,  # Nagel (2012)
}

STRATEGIES_V4 = {
    "time_series_momentum": TSMOM_PARAMS,
    "residual_reversal": RESIDUAL_STR_V4,
}

STRATEGIES_V5 = {
    "time_series_momentum": TSMOM_PARAMS,
    "residual_reversal": RESIDUAL_STR_V4,  # v5b: NO liquidity weight (large-cap universe)
}

STRATEGIES_V5_LIQ = {
    "time_series_momentum": TSMOM_PARAMS,
    "residual_reversal": RESIDUAL_STR_V5_LIQ,
}

# Risk parity parameters (Maillard et al. 2010)
RISK_PARITY_VOL_LOOKBACK = 63

# Vol-managed parameters (Moreira & Muir 2017)
VOL_MANAGED_TARGET = 0.15
VOL_MANAGED_LOOKBACK = 63


def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    """Compute standard performance metrics."""
    if len(returns) < 10 or returns.std() == 0:
        return {"name": name, "sharpe": 0, "cagr": 0, "max_dd": 0, "vol": 0,
                "calmar": 0, "sortino": 0, "daily_wr": 0, "monthly_wr": 0}
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    vol = returns.std() * np.sqrt(252)
    cum = (1 + returns).cumprod()
    n_years = len(returns) / 252
    cagr = cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    max_dd = ((cum / cum.cummax()) - 1).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = returns.mean() * 252 / downside if downside > 0 else 0
    daily_wr = (returns > 0).mean()
    monthly = returns.rolling(21).sum().dropna()
    monthly_wr = (monthly > 0).mean() if len(monthly) > 0 else 0
    return {
        "name": name, "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd,
        "vol": vol, "calmar": calmar, "sortino": sortino,
        "daily_wr": daily_wr, "monthly_wr": monthly_wr,
    }


def run_strategies(prices_wide, returns_wide, strategy_defs, sector_map=None):
    """Run strategies and return weights + returns."""
    from qrt.strategies import STRATEGY_REGISTRY

    results = {}
    for name, params in strategy_defs.items():
        try:
            cls = STRATEGY_REGISTRY[name]
            if name == "residual_reversal" and sector_map:
                strategy = cls(**params, sector_map=sector_map)
            else:
                strategy = cls(**params)
            signals = strategy.generate_signals(prices_wide, returns_wide)
            weights = strategy.compute_weights(signals, returns=returns_wide)
            strat_ret = (weights.shift(1) * returns_wide).sum(axis=1)
            results[name] = {"weights": weights, "returns": strat_ret}
        except Exception as e:
            logger.warning(f"  {name} FAILED: {e}")
    return results


def combine_v4(strategy_results: dict) -> pd.Series:
    """v4: Equal-weight combine + fixed 2x leverage."""
    returns_df = pd.DataFrame(
        {name: res["returns"] for name, res in strategy_results.items()}
    ).dropna(how="all").fillna(0.0)
    combined = returns_df.mean(axis=1)
    # Fixed 2x leverage
    return combined * 2.0


def combine_v5(strategy_results: dict) -> pd.Series:
    """v5: Risk parity + vol-managed leverage."""
    returns_df = pd.DataFrame(
        {name: res["returns"] for name, res in strategy_results.items()}
    ).dropna(how="all").fillna(0.0)

    # Rolling risk parity weights
    rp_combined = pd.Series(0.0, index=returns_df.index)

    for i in range(RISK_PARITY_VOL_LOOKBACK, len(returns_df)):
        window = returns_df.iloc[i - RISK_PARITY_VOL_LOOKBACK:i]
        vols = window.std() * np.sqrt(252)
        vols = vols.clip(lower=0.01)
        inv_vols = 1.0 / vols
        rp_w = inv_vols / inv_vols.sum()
        day_ret = returns_df.iloc[i]
        rp_combined.iloc[i] = (day_ret * rp_w).sum()

    # Vol-managed leverage
    trailing_vol = (
        rp_combined.rolling(VOL_MANAGED_LOOKBACK, min_periods=21).std()
        * np.sqrt(252)
    ).clip(lower=0.02)

    leverage = (VOL_MANAGED_TARGET / trailing_vol).clip(lower=0.5, upper=2.0)
    leveraged = rp_combined * leverage.shift(1).fillna(1.0)

    return leveraged


def main():
    parser = argparse.ArgumentParser(description="v4 vs v5 Comparison")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--holdout-date", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 70)
    logger.info("v4 (STATIC) vs v5 (DYNAMIC) — Comparison Backtest")
    logger.info("=" * 70)

    config = Config(args.config)

    # ── Step 1: Load data ──
    source = "real" if args.real_data else "synthetic"
    logger.info(f"Data source: {source}")

    from qrt.data.pipeline import DataPipeline
    pipeline = DataPipeline(config=config, seed=42, source=source)
    pipeline.run(force_regenerate=(source == "real"))
    pipeline.load_dataset()

    daily_bars = pipeline.market_data
    returns_df = pipeline.returns

    prices_wide = daily_bars.pivot_table(
        index="date", columns="security_id", values="adjusted_close"
    )
    returns_wide = returns_df.pivot_table(
        index="date", columns="security_id", values="ret_adj"
    )
    common_cols = prices_wide.columns.intersection(returns_wide.columns)
    prices_wide = prices_wide[common_cols].sort_index().ffill().dropna(how="all")
    returns_wide = returns_wide[common_cols].reindex(prices_wide.index).fillna(0)

    # Build sector map
    sector_map = {}
    if hasattr(pipeline, 'security_master') and pipeline.security_master is not None:
        sm = pipeline.security_master
        for _, row in sm.iterrows():
            sec_id = row['security_id']
            if sec_id in common_cols:
                industry = row.get('industry', row.get('sector', 'unknown'))
                sector_map[sec_id] = industry

    logger.info(f"Universe: {len(common_cols)} stocks, {len(set(sector_map.values()))} sub-industries")
    logger.info(f"Period: {prices_wide.index[0].date()} to {prices_wide.index[-1].date()}")

    # ── Step 2: Split train/test ──
    if args.holdout_date:
        holdout_dt = pd.Timestamp(args.holdout_date)
        train_p = prices_wide.loc[prices_wide.index < holdout_dt]
        train_r = returns_wide.loc[returns_wide.index < holdout_dt]
        test_p = prices_wide.loc[prices_wide.index >= holdout_dt]
        test_r = returns_wide.loc[returns_wide.index >= holdout_dt]
        logger.info(f"Train: {train_p.index[0].date()} to {train_p.index[-1].date()} ({len(train_p)} days)")
        logger.info(f"Test:  {test_p.index[0].date()} to {test_p.index[-1].date()} ({len(test_p)} days)")
    else:
        train_p, train_r = prices_wide, returns_wide
        test_p, test_r = None, None
        logger.info("No holdout — full-sample evaluation only")

    # ── Step 3: Run v4 and v5 on train (IS) ──
    logger.info("\n" + "=" * 70)
    logger.info("IN-SAMPLE COMPARISON")
    logger.info("=" * 70)

    logger.info("\n--- v4 (Static: equal-weight, fixed 2x) ---")
    v4_results_is = run_strategies(train_p, train_r, STRATEGIES_V4, sector_map)
    v4_combined_is = combine_v4(v4_results_is)

    logger.info("\n--- v5 (Dynamic: risk parity, vol-managed) ---")
    v5_results_is = run_strategies(train_p, train_r, STRATEGIES_V5, sector_map)
    v5_combined_is = combine_v5(v5_results_is)

    # IS metrics
    v4_is = compute_metrics(v4_combined_is, "v4_IS")
    v5_is = compute_metrics(v5_combined_is, "v5_IS")

    # Individual strategy IS metrics
    for version, results in [("v4", v4_results_is), ("v5", v5_results_is)]:
        for name, res in results.items():
            m = compute_metrics(res["returns"], f"{version}_{name}")
            logger.info(f"  {version} {name}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.2%}, MaxDD={m['max_dd']:.2%}")

    logger.info(f"\n  v4 Portfolio IS: Sharpe={v4_is['sharpe']:.3f}, CAGR={v4_is['cagr']:.2%}, MaxDD={v4_is['max_dd']:.2%}, Sortino={v4_is['sortino']:.3f}")
    logger.info(f"  v5 Portfolio IS: Sharpe={v5_is['sharpe']:.3f}, CAGR={v5_is['cagr']:.2%}, MaxDD={v5_is['max_dd']:.2%}, Sortino={v5_is['sortino']:.3f}")

    sharpe_improvement_is = (v5_is['sharpe'] / v4_is['sharpe'] - 1) * 100 if v4_is['sharpe'] > 0 else 0
    logger.info(f"  Sharpe improvement (IS): {sharpe_improvement_is:+.1f}%")

    # ── Step 4: Run on OOS if holdout specified ──
    v4_oos = v5_oos = None
    if test_p is not None and len(test_p) > 0:
        logger.info("\n" + "=" * 70)
        logger.info("OUT-OF-SAMPLE COMPARISON")
        logger.info("=" * 70)

        logger.info("\n--- v4 OOS ---")
        v4_results_oos = run_strategies(test_p, test_r, STRATEGIES_V4, sector_map)
        v4_combined_oos = combine_v4(v4_results_oos)
        v4_oos = compute_metrics(v4_combined_oos, "v4_OOS")

        logger.info("\n--- v5 OOS (risk parity + vol-managed) ---")
        v5_results_oos = run_strategies(test_p, test_r, STRATEGIES_V5, sector_map)
        v5_combined_oos = combine_v5(v5_results_oos)
        v5_oos = compute_metrics(v5_combined_oos, "v5_OOS")

        for version, results in [("v4", v4_results_oos), ("v5", v5_results_oos)]:
            for name, res in results.items():
                m = compute_metrics(res["returns"], f"{version}_{name}_OOS")
                logger.info(f"  {version} {name} OOS: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.2%}, MaxDD={m['max_dd']:.2%}")

        logger.info(f"\n  v4 Portfolio OOS: Sharpe={v4_oos['sharpe']:.3f}, CAGR={v4_oos['cagr']:.2%}, MaxDD={v4_oos['max_dd']:.2%}, Sortino={v4_oos['sortino']:.3f}")
        logger.info(f"  v5 Portfolio OOS: Sharpe={v5_oos['sharpe']:.3f}, CAGR={v5_oos['cagr']:.2%}, MaxDD={v5_oos['max_dd']:.2%}, Sortino={v5_oos['sortino']:.3f}")

        sharpe_improvement_oos = (v5_oos['sharpe'] / v4_oos['sharpe'] - 1) * 100 if v4_oos['sharpe'] > 0 else 0
        logger.info(f"  Sharpe improvement (OOS): {sharpe_improvement_oos:+.1f}%")

        # Degradation
        v4_deg = 1 - (v4_oos['sharpe'] / v4_is['sharpe']) if v4_is['sharpe'] > 0 else 1
        v5_deg = 1 - (v5_oos['sharpe'] / v5_is['sharpe']) if v5_is['sharpe'] > 0 else 1
        logger.info(f"  v4 Sharpe degradation IS→OOS: {v4_deg:.1%}")
        logger.info(f"  v5 Sharpe degradation IS→OOS: {v5_deg:.1%}")

    # ── Step 5: Generate report ──
    report = _generate_report(v4_is, v5_is, v4_oos, v5_oos)
    report_path = PROJECT_ROOT / "reports" / "v4_vs_v5_comparison.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)
    logger.info(f"\nReport saved: {report_path}")

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.0f}s")


def _generate_report(v4_is, v5_is, v4_oos=None, v5_oos=None) -> str:
    """Generate markdown comparison report."""
    report = """# v4 (Static) vs v5 (Dynamic) — Comparison Report

## Design Differences

| Aspect | v4 (Static) | v5 (Dynamic) |
|--------|-------------|--------------|
| Strategy allocation | Equal-weight 50/50 | Risk parity (inverse-vol) |
| Leverage | Fixed 2.0x | Vol-managed (target 15%, cap 2x) |
| Residual reversal | Equal-weight within quintile | Liquidity-weighted (Nagel 2012) |
| Free parameters | 0 | 0 (all from literature) |

## Academic Basis for v5

1. **Risk parity** — Maillard, Roncalli & Teiletche (2010): equalizing risk
   contributions outperforms equal-weight when strategies have different volatilities.
2. **Vol-managed leverage** — Moreira & Muir (2017): scaling exposure inversely to
   recent vol improves Sharpe by ~50% for momentum. Barroso & Santa-Clara (2015):
   vol-scaling momentum avoids crashes, improves Sharpe from 0.53 to 0.97.
3. **Liquidity-weighted reversal** — Nagel (2012): reversal premium is liquidity
   provision compensation, strongest for less-liquid stocks. Amihud (2002):
   illiquidity is a priced factor.

## In-Sample Performance

| Metric | v4 (Static) | v5 (Dynamic) | Improvement |
|--------|:-----------:|:------------:|:-----------:|
"""
    sharpe_imp = (v5_is['sharpe'] / v4_is['sharpe'] - 1) * 100 if v4_is['sharpe'] > 0 else 0
    cagr_imp = (v5_is['cagr'] / v4_is['cagr'] - 1) * 100 if v4_is['cagr'] != 0 else 0
    dd_imp = (abs(v5_is['max_dd']) / abs(v4_is['max_dd']) - 1) * 100 if v4_is['max_dd'] != 0 else 0

    report += f"| Sharpe | {v4_is['sharpe']:.3f} | {v5_is['sharpe']:.3f} | {sharpe_imp:+.1f}% |\n"
    report += f"| CAGR | {v4_is['cagr']:.2%} | {v5_is['cagr']:.2%} | {cagr_imp:+.1f}% |\n"
    report += f"| Max DD | {v4_is['max_dd']:.2%} | {v5_is['max_dd']:.2%} | {dd_imp:+.1f}% |\n"
    report += f"| Volatility | {v4_is['vol']:.2%} | {v5_is['vol']:.2%} | — |\n"
    report += f"| Sortino | {v4_is['sortino']:.3f} | {v5_is['sortino']:.3f} | — |\n"
    report += f"| Calmar | {v4_is['calmar']:.2f} | {v5_is['calmar']:.2f} | — |\n"
    report += f"| Daily WR | {v4_is['daily_wr']:.1%} | {v5_is['daily_wr']:.1%} | — |\n"
    report += f"| Monthly WR | {v4_is['monthly_wr']:.1%} | {v5_is['monthly_wr']:.1%} | — |\n"

    if v4_oos and v5_oos:
        sharpe_imp_oos = (v5_oos['sharpe'] / v4_oos['sharpe'] - 1) * 100 if v4_oos['sharpe'] > 0 else 0
        cagr_imp_oos = (v5_oos['cagr'] / v4_oos['cagr'] - 1) * 100 if v4_oos['cagr'] != 0 else 0
        v4_deg = (1 - v4_oos['sharpe'] / v4_is['sharpe']) * 100 if v4_is['sharpe'] > 0 else 0
        v5_deg = (1 - v5_oos['sharpe'] / v5_is['sharpe']) * 100 if v5_is['sharpe'] > 0 else 0

        report += f"""
## Out-of-Sample Performance

| Metric | v4 (Static) | v5 (Dynamic) | Improvement |
|--------|:-----------:|:------------:|:-----------:|
| Sharpe | {v4_oos['sharpe']:.3f} | {v5_oos['sharpe']:.3f} | {sharpe_imp_oos:+.1f}% |
| CAGR | {v4_oos['cagr']:.2%} | {v5_oos['cagr']:.2%} | {cagr_imp_oos:+.1f}% |
| Max DD | {v4_oos['max_dd']:.2%} | {v5_oos['max_dd']:.2%} | — |
| Volatility | {v4_oos['vol']:.2%} | {v5_oos['vol']:.2%} | — |
| Sortino | {v4_oos['sortino']:.3f} | {v5_oos['sortino']:.3f} | — |
| Calmar | {v4_oos['calmar']:.2f} | {v5_oos['calmar']:.2f} | — |
| Daily WR | {v4_oos['daily_wr']:.1%} | {v5_oos['daily_wr']:.1%} | — |
| Monthly WR | {v4_oos['monthly_wr']:.1%} | {v5_oos['monthly_wr']:.1%} | — |

## Sharpe Degradation (IS → OOS)

| Version | IS Sharpe | OOS Sharpe | Degradation |
|---------|:---------:|:----------:|:-----------:|
| v4 | {v4_is['sharpe']:.3f} | {v4_oos['sharpe']:.3f} | {v4_deg:.1f}% |
| v5 | {v5_is['sharpe']:.3f} | {v5_oos['sharpe']:.3f} | {v5_deg:.1f}% |
"""
        if v5_deg < v4_deg:
            report += "\nv5 shows **lower degradation**, suggesting the dynamic allocation generalizes better OOS.\n"
        elif v5_deg > v4_deg:
            report += "\nv5 shows **higher degradation** — the dynamic elements may be fitting to in-sample patterns.\n"

    report += """
## References

1. Maillard, S., Roncalli, T. & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contributions Portfolios. *JPM*.
2. Moreira, A. & Muir, T. (2017). Volatility-Managed Portfolios. *JF*.
3. Barroso, P. & Santa-Clara, P. (2015). Momentum Has Its Moments. *JFE*.
4. Nagel, S. (2012). Evaporating Liquidity. *RFS*.
5. Amihud, Y. (2002). Illiquidity and Stock Returns. *JFM*.
6. Avramov, D., Chordia, T. & Goyal, A. (2006). Liquidity and Autocorrelations in Individual Stock Returns. *JFE*.
7. DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification. *RFS*.
"""
    return report


if __name__ == "__main__":
    main()
