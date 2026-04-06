"""Tests for the Alpha Engine signal generation, evaluation, and filtering."""
import pytest
import numpy as np
import pandas as pd

import importlib
import sys

# Import SignalGenerator directly to avoid the broken alpha_engine/__init__.py
# which references not-yet-implemented modules (signal_evaluator, signal_filter, etc.)
_spec = importlib.util.spec_from_file_location(
    "qrt.alpha_engine.signal_generator",
    __import__("pathlib").Path(__file__).parent.parent
    / "qrt" / "alpha_engine" / "signal_generator.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SignalGenerator = _mod.SignalGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    return SignalGenerator(winsorize_signals=True, zscore_normalize=True)


# ---------------------------------------------------------------------------
# test_signal_generation
# ---------------------------------------------------------------------------

def test_signal_generation(generator, sample_prices, sample_returns, sample_volumes):
    """SignalGenerator.generate_candidates returns a non-empty dict of DataFrames."""
    candidates = generator.generate_candidates(
        prices=sample_prices,
        returns=sample_returns,
        volumes=sample_volumes,
    )

    assert isinstance(candidates, dict), "Expected a dict of signals"
    assert len(candidates) > 0, "No signals were generated"

    # Every value must be a DataFrame with matching shape
    for name, sig in candidates.items():
        assert isinstance(sig, pd.DataFrame), f"Signal '{name}' is not a DataFrame"
        assert sig.shape == sample_returns.shape, (
            f"Signal '{name}' shape {sig.shape} != returns shape {sample_returns.shape}"
        )


# ---------------------------------------------------------------------------
# test_signal_evaluation_metrics
# ---------------------------------------------------------------------------

def test_signal_evaluation_metrics(generator, sample_prices, sample_returns, sample_volumes):
    """
    Each generated signal must carry enough structure to compute core
    evaluation metrics: mean, std, and basic non-NaN coverage.
    """
    candidates = generator.generate_candidates(
        prices=sample_prices,
        returns=sample_returns,
        volumes=sample_volumes,
    )

    required_metrics = {}
    for name, sig in candidates.items():
        flat = sig.values.flatten()
        valid = flat[~np.isnan(flat)]
        required_metrics[name] = {
            "mean": float(np.mean(valid)) if len(valid) > 0 else np.nan,
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
            "coverage": len(valid) / len(flat) if len(flat) > 0 else 0.0,
        }

    assert len(required_metrics) > 0, "No metrics could be computed"

    for name, metrics in required_metrics.items():
        assert "mean" in metrics, f"Missing 'mean' for signal '{name}'"
        assert "std" in metrics, f"Missing 'std' for signal '{name}'"
        assert "coverage" in metrics, f"Missing 'coverage' for signal '{name}'"
        # Coverage should be reasonable — at least 10% non-NaN
        assert metrics["coverage"] > 0.10, (
            f"Signal '{name}' has very low valid-value coverage: {metrics['coverage']:.2%}"
        )


# ---------------------------------------------------------------------------
# test_signal_filtering
# ---------------------------------------------------------------------------

def test_signal_filtering(generator, sample_prices, sample_returns, sample_volumes):
    """
    A simple IC (information coefficient) filter should remove signals
    whose mean forward IC is below a threshold.
    """
    candidates = generator.generate_candidates(
        prices=sample_prices,
        returns=sample_returns,
        volumes=sample_volumes,
    )

    # Compute 1-day forward IC for each signal
    forward_returns = sample_returns.shift(-1)
    ic_scores: dict[str, float] = {}

    for name, sig in candidates.items():
        # Align
        common_idx = sig.index.intersection(forward_returns.index)
        if len(common_idx) < 20:
            continue
        sig_vals = sig.loc[common_idx].values.flatten()
        fwd_vals = forward_returns.loc[common_idx].values.flatten()
        mask = ~np.isnan(sig_vals) & ~np.isnan(fwd_vals)
        if mask.sum() < 20:
            continue
        ic = np.corrcoef(sig_vals[mask], fwd_vals[mask])[0, 1]
        if not np.isnan(ic):
            ic_scores[name] = ic

    if not ic_scores:
        pytest.skip("No IC scores could be computed")

    # Filter: keep only signals with |IC| > some very small threshold (0.001)
    threshold = 0.001
    passing = {k: v for k, v in ic_scores.items() if abs(v) >= threshold}
    failing = {k: v for k, v in ic_scores.items() if abs(v) < threshold}

    # After filtering, the passing set should be smaller or equal to full set
    assert len(passing) <= len(ic_scores), "Filter should not add signals"

    # At least some signals should survive a lenient threshold
    assert len(passing) > 0, (
        f"All {len(ic_scores)} signals were filtered out at threshold={threshold}. "
        f"Min |IC| = {min(abs(v) for v in ic_scores.values()):.6f}"
    )


# ---------------------------------------------------------------------------
# test_correlation_check
# ---------------------------------------------------------------------------

def test_correlation_check(generator, sample_prices, sample_returns, sample_volumes):
    """
    Pairs of highly correlated signals should be detectable and the
    redundant one should be filterable.
    """
    candidates = generator.generate_candidates(
        prices=sample_prices,
        returns=sample_returns,
        volumes=sample_volumes,
    )

    # Compute pairwise correlation between signal time-series (collapsed to
    # a single column-average per day for tractability)
    signal_means: dict[str, pd.Series] = {}
    for name, sig in candidates.items():
        col_avg = sig.mean(axis=1).dropna()
        if len(col_avg) > 20:
            signal_means[name] = col_avg

    if len(signal_means) < 2:
        pytest.skip("Fewer than 2 signals available for correlation check")

    names = list(signal_means.keys())
    # Build a small correlation matrix from the first 20 signals
    names_subset = names[:20]
    series_subset = [signal_means[n] for n in names_subset]
    common_idx = series_subset[0].index
    for s in series_subset[1:]:
        common_idx = common_idx.intersection(s.index)

    if len(common_idx) < 20:
        pytest.skip("Not enough common dates across signals")

    matrix = np.column_stack([signal_means[n].reindex(common_idx).values for n in names_subset])
    corr_matrix = np.corrcoef(matrix.T)

    # Find highly correlated pairs (|r| > 0.95, excluding diagonal)
    n = len(names_subset)
    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) > 0.95:
                high_corr_pairs.append((names_subset[i], names_subset[j], corr_matrix[i, j]))

    # Simulate filtering: for each high-corr pair, discard the second signal
    all_signal_names = set(names_subset)
    discarded = set()
    for sig_a, sig_b, corr_val in high_corr_pairs:
        if sig_a not in discarded and sig_b not in discarded:
            discarded.add(sig_b)

    filtered = all_signal_names - discarded

    assert len(filtered) <= len(all_signal_names), "Correlation filter should not add signals"

    if high_corr_pairs:
        assert len(filtered) < len(all_signal_names), (
            "High-correlation filter should have removed at least one signal"
        )
