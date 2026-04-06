"""
Monte Carlo Risk Simulation Module

Implements block bootstrap, return permutation, correlation perturbation,
and leverage stress testing for portfolio risk analysis. Pure risk analysis —
does NOT modify backtest results.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


class MonteCarloRiskSimulator:
    """Monte Carlo risk simulator for portfolio return streams.

    Parameters
    ----------
    n_simulations : int
        Number of simulation paths for bootstrap (default 5000).
    block_size : int
        Block length for block bootstrap, should be 5–10 (default 5).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_simulations: int = 5000, block_size: int = 5, random_state: int = 42):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if n_simulations < 1:
            raise ValueError("n_simulations must be >= 1")

        self.n_simulations = n_simulations
        self.block_size = block_size
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_bootstrap(self, returns: pd.Series) -> dict:
        """Block bootstrap simulation preserving volatility clustering.

        Samples contiguous blocks of ``block_size`` returns (with replacement)
        and stitches them together to form synthetic return series of the same
        length as the original.

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns (not log returns).

        Returns
        -------
        dict
            ``simulated_paths`` – DataFrame (days x simulations) of cumulative
            wealth curves starting at 1.0.
            ``risk_metrics`` – DataFrame with one row per simulation.
        """
        ret_arr = np.asarray(returns, dtype=np.float64)
        n = len(ret_arr)
        if n == 0:
            raise ValueError("returns series is empty")

        bs = self.block_size
        # Number of blocks needed to cover the full length
        n_blocks = int(np.ceil(n / bs))

        # Maximum valid start index for a block
        max_start = n - bs  # inclusive

        # Pre-draw all block start indices: shape (n_simulations, n_blocks)
        starts = self._rng.randint(0, max_start + 1, size=(self.n_simulations, n_blocks))

        # Build simulated return matrix in a memory-friendly way.
        # Each row of `starts` gives the block-start indices for one simulation.
        # We construct a flat index array per simulation then slice returns.
        block_offsets = np.arange(bs)  # [0, 1, ..., bs-1]

        # shape (n_simulations, n_blocks, bs) -> flatten last two dims -> truncate to n
        indices = (starts[:, :, np.newaxis] + block_offsets[np.newaxis, np.newaxis, :]).reshape(
            self.n_simulations, n_blocks * bs
        )[:, :n]

        # Gather returns — shape (n_simulations, n)
        sim_returns = ret_arr[indices]

        # Cumulative wealth curves (start at 1.0)
        sim_paths = np.cumprod(1.0 + sim_returns, axis=1)

        paths_df = pd.DataFrame(sim_paths.T, columns=[f"sim_{i}" for i in range(self.n_simulations)])
        metrics_df = self.compute_risk_metrics(paths_df)

        return {"simulated_paths": paths_df, "risk_metrics": metrics_df}

    def run_permutation(self, returns: pd.Series, n_simulations: int = 2000) -> dict:
        """Random return permutation (sequence-risk test).

        Shuffles the order of daily returns to destroy any serial
        dependence while keeping the marginal distribution intact.

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns.
        n_simulations : int
            Number of shuffled paths (default 2000).

        Returns
        -------
        dict
            ``simulated_paths`` and ``risk_metrics``.
        """
        ret_arr = np.asarray(returns, dtype=np.float64)
        n = len(ret_arr)
        if n == 0:
            raise ValueError("returns series is empty")

        # Shuffle by generating random sort keys — vectorised, no Python loop
        sort_keys = self._rng.random((n_simulations, n))
        order = np.argsort(sort_keys, axis=1)
        sim_returns = ret_arr[order]

        sim_paths = np.cumprod(1.0 + sim_returns, axis=1)

        paths_df = pd.DataFrame(sim_paths.T, columns=[f"sim_{i}" for i in range(n_simulations)])
        metrics_df = self.compute_risk_metrics(paths_df)

        return {"simulated_paths": paths_df, "risk_metrics": metrics_df}

    def run_correlation_stress(
        self,
        strategy_returns: dict[str, pd.Series],
        weights: pd.Series,
    ) -> dict:
        """Correlation perturbation stress test.

        1. Estimate the empirical correlation matrix of the strategies.
        2. For each simulation, perturb the correlation matrix by adding
           Gaussian noise, then project back to the nearest valid (PSD)
           correlation matrix.
        3. Simulate correlated returns from the perturbed matrix (using
           the empirical marginal means and standard deviations).
        4. Compute portfolio returns with the given weights and derive
           risk metrics.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Mapping of strategy name -> daily returns series.
        weights : pd.Series
            Strategy weights (index = strategy names, values sum to ~1).

        Returns
        -------
        dict
            ``simulated_paths``, ``risk_metrics``, ``perturbed_correlations``
            (list of correlation matrices used).
        """
        names = list(strategy_returns.keys())
        # Align all series to a common index (inner join)
        df = pd.DataFrame(strategy_returns).dropna()
        if df.empty:
            raise ValueError("No overlapping data across strategies")

        n_days = len(df)
        n_strats = len(names)

        ret_matrix = df.values.astype(np.float64)  # (n_days, n_strats)
        means = ret_matrix.mean(axis=0)
        stds = ret_matrix.std(axis=0, ddof=1)
        corr = np.corrcoef(ret_matrix, rowvar=False)

        # Align weights to the same order
        w = np.array([float(weights.get(name, 0.0)) for name in names])

        perturbed_corrs = []
        all_port_paths = np.empty((self.n_simulations, n_days))

        for i in range(self.n_simulations):
            # Perturb correlation: add symmetric noise, project to nearest PSD corr
            noise = self._rng.normal(0, 0.05, size=(n_strats, n_strats))
            noise = (noise + noise.T) / 2.0
            np.fill_diagonal(noise, 0.0)
            perturbed = corr + noise
            perturbed = self._nearest_corr(perturbed)
            perturbed_corrs.append(perturbed)

            # Cholesky factor of perturbed correlation
            try:
                L = np.linalg.cholesky(perturbed)
            except np.linalg.LinAlgError:
                # Fallback: use original correlation
                L = np.linalg.cholesky(corr)

            # Generate correlated standard normals -> scale to marginal dist
            z = self._rng.standard_normal((n_days, n_strats))
            correlated = z @ L.T  # (n_days, n_strats)
            sim_rets = correlated * stds[np.newaxis, :] + means[np.newaxis, :]

            # Portfolio returns
            port_rets = sim_rets @ w
            all_port_paths[i, :] = np.cumprod(1.0 + port_rets)

        paths_df = pd.DataFrame(
            all_port_paths.T, columns=[f"sim_{i}" for i in range(self.n_simulations)]
        )
        metrics_df = self.compute_risk_metrics(paths_df)

        return {
            "simulated_paths": paths_df,
            "risk_metrics": metrics_df,
            "perturbed_correlations": perturbed_corrs,
        }

    def run_leverage_stress(
        self,
        returns: pd.Series,
        leverage_levels: list[float] | None = None,
    ) -> dict:
        """Kelly / leverage stress test.

        For each leverage level, scale returns and compute risk metrics over
        block-bootstrap paths.

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns.
        leverage_levels : list[float]
            Leverage multipliers to test (default [0.5, 1.0, 1.5, 2.0]).

        Returns
        -------
        dict
            Mapping leverage_level -> {simulated_paths, risk_metrics}.
            Also includes a ``summary`` DataFrame comparing metrics across
            leverage levels.
        """
        if leverage_levels is None:
            leverage_levels = [0.5, 1.0, 1.5, 2.0]

        ret_arr = np.asarray(returns, dtype=np.float64)

        results: dict = {}
        summary_rows = []

        for lev in leverage_levels:
            leveraged = ret_arr * lev
            leveraged_series = pd.Series(leveraged, index=returns.index if hasattr(returns, "index") else None)

            # Re-use bootstrap machinery (save the RNG state so each leverage
            # level sees the *same* block draws — apples-to-apples comparison).
            rng_state = self._rng.get_state()
            res = self.run_bootstrap(leveraged_series)
            self._rng.set_state(rng_state)  # reset so next level draws same blocks

            results[lev] = res

            # Aggregate summary
            m = res["risk_metrics"]
            summary_rows.append(
                {
                    "leverage": lev,
                    "median_sharpe": m["sharpe"].median(),
                    "median_cagr": m["cagr"].median(),
                    "median_max_drawdown": m["max_drawdown"].median(),
                    "median_volatility": m["volatility"].median(),
                    "probability_of_ruin": m["probability_of_ruin"].mean(),
                    "mean_terminal_wealth": m["terminal_wealth"].mean(),
                    "p5_terminal_wealth": m["terminal_wealth"].quantile(0.05),
                }
            )

        results["summary"] = pd.DataFrame(summary_rows)
        return results

    def compute_risk_metrics(self, simulated_paths: pd.DataFrame) -> pd.DataFrame:
        """Compute risk metrics for each simulation path.

        Parameters
        ----------
        simulated_paths : pd.DataFrame
            Each column is a cumulative wealth curve starting at 1.0
            (or the first value of the cum-product). Rows = time steps.

        Returns
        -------
        pd.DataFrame
            One row per simulation with columns: max_drawdown, sharpe, cagr,
            volatility, skew, kurtosis, terminal_wealth, probability_of_ruin.
        """
        paths = simulated_paths.values.astype(np.float64)  # (n_days, n_sims)
        n_days, n_sims = paths.shape

        # Daily returns from equity curves
        daily_rets = np.empty_like(paths)
        daily_rets[0, :] = paths[0, :] - 1.0  # first day return relative to 1.0
        daily_rets[1:, :] = paths[1:, :] / paths[:-1, :] - 1.0

        # --- Max drawdown ---
        running_max = np.maximum.accumulate(paths, axis=0)
        drawdowns = (paths - running_max) / running_max
        max_dd = drawdowns.min(axis=0)  # most negative value per sim

        # --- Annualised volatility ---
        vol = np.std(daily_rets, axis=0, ddof=1) * np.sqrt(252)

        # --- Annualised mean return (for Sharpe) ---
        mean_daily = np.mean(daily_rets, axis=0)
        ann_mean = mean_daily * 252

        # --- Sharpe (risk-free = 0) ---
        std_daily = np.std(daily_rets, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.where(std_daily > 0, (mean_daily / std_daily) * np.sqrt(252), 0.0)

        # --- CAGR ---
        terminal = paths[-1, :]
        years = n_days / 252.0
        with np.errstate(divide="ignore", invalid="ignore"):
            cagr = np.where(
                terminal > 0,
                np.sign(terminal) * (np.abs(terminal) ** (1.0 / years)) - 1.0,
                -1.0,
            )

        # --- Skew & Kurtosis ---
        skew = np.empty(n_sims)
        kurt = np.empty(n_sims)
        for j in range(n_sims):
            col = daily_rets[:, j]
            skew[j] = scipy_stats.skew(col, bias=False)
            kurt[j] = scipy_stats.kurtosis(col, bias=False)  # excess kurtosis

        # --- Probability of ruin (equity falls below 50% of starting) ---
        min_equity = paths.min(axis=0)
        ruin = (min_equity < 0.5).astype(np.float64)

        metrics = pd.DataFrame(
            {
                "max_drawdown": max_dd,
                "sharpe": sharpe,
                "cagr": cagr,
                "volatility": vol,
                "skew": skew,
                "kurtosis": kurt,
                "terminal_wealth": terminal,
                "probability_of_ruin": ruin,
            }
        )
        return metrics

    def run_full_analysis(
        self,
        portfolio_returns: pd.Series,
        strategy_returns: dict[str, pd.Series] | None = None,
        weights: pd.Series | None = None,
    ) -> dict:
        """Run all Monte Carlo analyses and return combined results.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Aggregate portfolio daily returns.
        strategy_returns : dict[str, pd.Series], optional
            Per-strategy returns for correlation stress test.
        weights : pd.Series, optional
            Strategy weights (required if strategy_returns is provided).

        Returns
        -------
        dict
            Keys: ``bootstrap``, ``permutation``, ``leverage_stress``,
            and optionally ``correlation_stress``.  Each value is the dict
            returned by the corresponding ``run_*`` method.
            Also includes ``summary`` with high-level statistics.
        """
        results: dict = {}

        # 1. Block bootstrap
        results["bootstrap"] = self.run_bootstrap(portfolio_returns)

        # 2. Permutation
        results["permutation"] = self.run_permutation(portfolio_returns)

        # 3. Leverage stress
        results["leverage_stress"] = self.run_leverage_stress(portfolio_returns)

        # 4. Correlation stress (optional)
        if strategy_returns is not None and weights is not None:
            results["correlation_stress"] = self.run_correlation_stress(strategy_returns, weights)

        # High-level summary
        bs_metrics = results["bootstrap"]["risk_metrics"]
        perm_metrics = results["permutation"]["risk_metrics"]

        results["summary"] = {
            "bootstrap": {
                "median_sharpe": float(bs_metrics["sharpe"].median()),
                "median_max_drawdown": float(bs_metrics["max_drawdown"].median()),
                "median_cagr": float(bs_metrics["cagr"].median()),
                "probability_of_ruin": float(bs_metrics["probability_of_ruin"].mean()),
                "p5_terminal_wealth": float(bs_metrics["terminal_wealth"].quantile(0.05)),
                "p50_terminal_wealth": float(bs_metrics["terminal_wealth"].quantile(0.50)),
                "p95_terminal_wealth": float(bs_metrics["terminal_wealth"].quantile(0.95)),
            },
            "permutation": {
                "median_sharpe": float(perm_metrics["sharpe"].median()),
                "median_max_drawdown": float(perm_metrics["max_drawdown"].median()),
                "median_cagr": float(perm_metrics["cagr"].median()),
                "probability_of_ruin": float(perm_metrics["probability_of_ruin"].mean()),
            },
            "leverage_optimal": self._find_optimal_leverage(results["leverage_stress"]),
        }

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_corr(A: np.ndarray) -> np.ndarray:
        """Project a symmetric matrix to the nearest valid correlation matrix.

        Uses the alternating projections method (Higham, 2002) — project
        onto the set of PSD matrices and onto the set of matrices with
        unit diagonal, alternating until convergence.
        """
        n = A.shape[0]
        Y = A.copy()
        delta_S = np.zeros_like(A)

        for _ in range(100):
            R = Y - delta_S

            # Project onto PSD cone
            eigvals, eigvecs = np.linalg.eigh(R)
            eigvals = np.maximum(eigvals, 1e-10)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T

            delta_S = X - R

            # Project onto unit-diagonal matrices
            Y = X.copy()
            np.fill_diagonal(Y, 1.0)

            # Symmetrise
            Y = (Y + Y.T) / 2.0

            # Check convergence
            if np.max(np.abs(Y - X)) < 1e-8:
                break

        # Clip to [-1, 1]
        np.clip(Y, -1.0, 1.0, out=Y)
        np.fill_diagonal(Y, 1.0)
        return Y

    @staticmethod
    def _find_optimal_leverage(leverage_results: dict) -> dict:
        """Identify the leverage level that maximises median risk-adjusted return.

        Uses a simple criterion: highest median Sharpe with ruin probability < 5%.
        """
        summary = leverage_results.get("summary")
        if summary is None or summary.empty:
            return {"optimal_leverage": 1.0, "reason": "no summary available"}

        viable = summary[summary["probability_of_ruin"] < 0.05]
        if viable.empty:
            # Fall back to lowest ruin probability
            best = summary.loc[summary["probability_of_ruin"].idxmin()]
            return {
                "optimal_leverage": float(best["leverage"]),
                "reason": "all leverage levels exceed 5% ruin probability; chose lowest ruin",
            }

        best = viable.loc[viable["median_sharpe"].idxmax()]
        return {
            "optimal_leverage": float(best["leverage"]),
            "reason": "highest median Sharpe with ruin probability < 5%",
        }
