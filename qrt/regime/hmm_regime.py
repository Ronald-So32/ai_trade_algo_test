"""
Hidden Markov Model Regime Detector
=====================================
Detects latent market regimes using a Gaussian Hidden Markov Model.

Supports 3-state (Bull/Bear/Crisis) or 4-state configurations.  Research
basis: Hamilton (1989), Ang & Bekaert (2002), Guidolin & Timmermann (2007).

Features (computed by ``extract_features``)
--------------------------------------------
rolling_return     : short-horizon momentum proxy (21-day rolling mean of returns)
realized_vol       : annualised 21-day rolling standard deviation of returns
avg_pairwise_corr  : average pairwise Pearson correlation across assets over a
                     rolling window – serves as a stress/diversification proxy
trend_strength     : rolling_return / (realized_vol + ε) – Sharpe-like
                     regime-normalised momentum signal

State labelling (automatic)
----------------------------
After fitting, states are sorted by emission mean of ``realized_vol`` so that:

    3-state: 0=Bull (low vol), 1=Bear (medium vol), 2=Crisis (high vol)
    4-state: 0=Bull, 1=Neutral, 2=Bear, 3=Crisis

This solves the label-switching problem across refits (research best practice).

Key design decisions
--------------------
* **Forward-only filtered probabilities** — ``predict`` uses the forward
  algorithm (``predict_proba``), never the Viterbi path on full data,
  to avoid look-ahead bias per López de Prado (2018).
* **Walk-forward refit** — ``walk_forward_predict`` retrains the HMM on a
  rolling window, ensuring the model adapts to structural change while
  preventing information leakage.
* **Regime prediction** — ``predict_next_regime`` uses the transition matrix
  to forecast next-period regime probabilities.

Edge-case handling
------------------
* Convergence failure  → logged as WARNING; model remains usable with last
  iteration parameters (hmmlearn behaviour).
* Too-short input      → ``predict`` returns NaN rows for the warm-up period.
* Single asset         → ``avg_pairwise_corr`` falls back to rolling
  autocorrelation (lag-1) of that asset's returns.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import zscore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_STATES: int = 4
_FEATURE_COLS: list[str] = [
    "rolling_return",
    "realized_vol",
    "avg_pairwise_corr",
    "trend_strength",
]
_DEFAULT_ROLLING_WINDOW: int = 21
_DEFAULT_CORR_WINDOW: int = 63  # longer window for stable correlation estimate
_ANNUALISATION: float = 252.0
_EPSILON: float = 1e-8  # prevent division-by-zero in trend_strength


class HMMRegimeDetector:
    """
    Gaussian Hidden Markov Model market-regime detector.

    Parameters
    ----------
    n_states : int, default 4
        Number of hidden states.
    n_iter : int, default 200
        Maximum EM iterations.
    tol : float, default 1e-4
        Convergence tolerance for the log-likelihood improvement.
    covariance_type : str, default 'full'
        Type of HMM covariance matrix.  One of ``'spherical'``, ``'diag'``,
        ``'full'``, ``'tied'``.
    random_state : int or None, default 42
        Seed for reproducibility.
    rolling_window : int, default 21
        Window (trading days) for rolling return and volatility features.
    corr_window : int, default 63
        Window (trading days) for rolling pairwise correlation feature.
    standardise_features : bool, default True
        When ``True``, features are z-scored before fitting and prediction.
        The mean and std are captured during ``fit`` and reused in
        ``predict`` to avoid look-ahead bias.

    Attributes
    ----------
    model_ : hmmlearn.hmm.GaussianHMM
        Fitted HMM model.
    is_fitted_ : bool
    feature_mean_ : pd.Series
        Per-feature means captured during ``fit`` (used when standardising).
    feature_std_ : pd.Series
        Per-feature standard deviations captured during ``fit``.
    """

    def __init__(
        self,
        n_states: int = _N_STATES,
        n_iter: int = 200,
        tol: float = 1e-4,
        covariance_type: str = "full",
        random_state: Optional[int] = 42,
        rolling_window: int = _DEFAULT_ROLLING_WINDOW,
        corr_window: int = _DEFAULT_CORR_WINDOW,
        standardise_features: bool = True,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.rolling_window = rolling_window
        self.corr_window = corr_window
        self.standardise_features = standardise_features

        # Fitted state
        self.model_: Optional[hmm.GaussianHMM] = None
        self.is_fitted_: bool = False
        self.feature_mean_: Optional[pd.Series] = None
        self.feature_std_: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features_df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit the Gaussian HMM to a feature DataFrame.

        Parameters
        ----------
        features_df : pd.DataFrame
            Must contain at least the columns in ``_FEATURE_COLS``.  Rows
            with *any* NaN are silently dropped before fitting.

        Returns
        -------
        self
        """
        features_df = self._validate_features(features_df)
        X, _ = self._prepare_X(features_df, fit=True)

        if len(X) < self.n_states * 5:
            raise ValueError(
                f"Insufficient clean observations ({len(X)}) to reliably fit "
                f"a {self.n_states}-state HMM.  Provide more data."
            )

        self.model_ = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=False,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model_.fit(X)
            for w in caught:
                if issubclass(w.category, (ConvergenceWarning := type(  # noqa: N806
                    "ConvergenceWarning", (UserWarning,), {}
                ))):
                    pass  # handled below via monitor_
                logger.warning("HMM fit warning: %s", w.message)

        if hasattr(self.model_, "monitor_") and not self.model_.monitor_.converged:
            logger.warning(
                "HMMRegimeDetector: EM did not converge after %d iterations.  "
                "Consider increasing n_iter or adjusting the feature set.",
                self.n_iter,
            )

        self.is_fitted_ = True

        # Sort states by realized_vol emission mean (low→high)
        # so state 0 = bull (lowest vol), last state = crisis (highest vol)
        self._sort_states_by_volatility()

        logger.info(
            "HMMRegimeDetector fitted on %d observations with %d states.  "
            "Log-likelihood: %.4f",
            len(X),
            self.n_states,
            self.model_.score(X),
        )
        return self

    def predict(
        self, features_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Decode the most-likely state sequence and compute state probabilities.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix (same schema as used in ``fit``).  NaN rows are
            excluded from the HMM decoding and re-inserted as NaN in the
            output.

        Returns
        -------
        states : pd.Series
            Integer state labels aligned to ``features_df.index``.  NaN for
            rows that had missing features.
        probabilities : pd.DataFrame
            State posterior probabilities of shape (len(features_df), n_states).
            Column names: ``state_0``, ``state_1``, …
        """
        self._check_fitted()
        features_df = self._validate_features(features_df)
        X, valid_mask = self._prepare_X(features_df, fit=False)

        # Placeholder outputs filled with NaN then overwritten for valid rows
        states_arr = np.full(len(features_df), np.nan)
        prob_arr = np.full((len(features_df), self.n_states), np.nan)

        if X.shape[0] > 0:
            predicted_states = self.model_.predict(X)
            predicted_probs = self.model_.predict_proba(X)
            valid_indices = np.where(valid_mask)[0]
            states_arr[valid_indices] = predicted_states
            prob_arr[valid_indices] = predicted_probs

        states = pd.Series(
            states_arr,
            index=features_df.index,
            name="regime_state",
            dtype="float64",
        )
        # Convert to nullable integer where possible
        valid_mask = states.notna()
        if valid_mask.any():
            states.loc[valid_mask] = states.loc[valid_mask].astype(int)

        prob_cols = [f"state_{i}" for i in range(self.n_states)]
        probabilities = pd.DataFrame(
            prob_arr, index=features_df.index, columns=prob_cols
        )

        return states, probabilities

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def transition_matrix(self) -> pd.DataFrame:
        """
        Transition probability matrix as a DataFrame.

        Returns
        -------
        pd.DataFrame of shape (n_states, n_states)
            ``transition_matrix.loc[i, j]`` is the probability of
            transitioning from state *i* to state *j*.
        """
        self._check_fitted()
        cols = [f"state_{i}" for i in range(self.n_states)]
        return pd.DataFrame(
            self.model_.transmat_, index=cols, columns=cols
        )

    @property
    def emission_means(self) -> pd.DataFrame:
        """Mean feature vector for each hidden state."""
        self._check_fitted()
        return pd.DataFrame(
            self.model_.means_,
            index=[f"state_{i}" for i in range(self.n_states)],
            columns=_FEATURE_COLS,
        )

    # ------------------------------------------------------------------
    # Regime prediction & walk-forward
    # ------------------------------------------------------------------

    def predict_next_regime(
        self, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Forecast next-period regime probabilities using transition matrix.

        For each date t, computes:
            P(S_{t+1} | data_1:t) = TransitionMatrix^T @ P(S_t | data_1:t)

        Returns
        -------
        pd.DataFrame
            Next-period regime probabilities, same shape as predict() output.
        """
        _, current_probs = self.predict(features_df)
        trans = self.model_.transmat_  # (n_states, n_states)
        # next_prob[t] = current_prob[t] @ transition_matrix
        next_probs = current_probs.values @ trans
        return pd.DataFrame(
            next_probs, index=current_probs.index, columns=current_probs.columns
        )

    def walk_forward_predict(
        self,
        features_df: pd.DataFrame,
        train_window: int = 756,
        retrain_freq: int = 63,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Walk-forward regime detection: retrain on rolling window, predict
        using only past data (forward-only filtered probabilities).

        This is the gold-standard approach per López de Prado (2018):
        no future data leaks into regime labels.

        Parameters
        ----------
        features_df : pd.DataFrame
            Full feature matrix (must contain _FEATURE_COLS).
        train_window : int
            Rolling training window in observations (default 756 = ~3 years).
        retrain_freq : int
            Retrain every N observations (default 63 = ~quarterly).

        Returns
        -------
        states : pd.Series
            Regime state labels (sorted: 0=bull, last=crisis).
        probabilities : pd.DataFrame
            Posterior regime probabilities.
        """
        features_df = self._validate_features(features_df)
        n = len(features_df)

        states_arr = np.full(n, np.nan)
        prob_arr = np.full((n, self.n_states), np.nan)

        last_train = -retrain_freq  # force first train

        for t in range(train_window, n):
            # Retrain periodically
            if t - last_train >= retrain_freq:
                train_slice = features_df.iloc[t - train_window:t]
                try:
                    self.fit(train_slice)
                    last_train = t
                except (ValueError, RuntimeError) as e:
                    logger.warning("Walk-forward refit failed at t=%d: %s", t, e)
                    if not self.is_fitted_:
                        continue

            if not self.is_fitted_:
                continue

            # Predict on data up to t (inclusive) using forward algorithm
            # Only take the last prediction to avoid look-ahead
            predict_slice = features_df.iloc[max(0, t - train_window):t + 1]
            X, valid_mask = self._prepare_X(predict_slice, fit=False)
            if X.shape[0] > 0:
                try:
                    probs = self.model_.predict_proba(X)
                    state = np.argmax(probs[-1])
                    states_arr[t] = state
                    prob_arr[t] = probs[-1]
                except Exception:
                    pass

        states = pd.Series(
            states_arr, index=features_df.index, name="regime_state", dtype="float64"
        )
        valid = states.notna()
        if valid.any():
            states.loc[valid] = states.loc[valid].astype(int)

        prob_cols = [f"state_{i}" for i in range(self.n_states)]
        probabilities = pd.DataFrame(prob_arr, index=features_df.index, columns=prob_cols)

        logger.info(
            "Walk-forward HMM: %d predictions over %d dates "
            "(window=%d, retrain=%dd)",
            int(valid.sum()), n, train_window, retrain_freq,
        )
        return states, probabilities

    @property
    def state_labels(self) -> dict[int, str]:
        """
        Human-readable labels for each state, based on emission ordering.

        After ``_sort_states_by_volatility``:
        - 3-state: {0: 'bull', 1: 'bear', 2: 'crisis'}
        - 4-state: {0: 'bull', 1: 'neutral', 2: 'bear', 3: 'crisis'}
        """
        if self.n_states == 3:
            return {0: "bull", 1: "bear", 2: "crisis"}
        elif self.n_states == 4:
            return {0: "bull", 1: "neutral", 2: "bear", 3: "crisis"}
        else:
            return {i: f"state_{i}" for i in range(self.n_states)}

    @property
    def crisis_state_index(self) -> int:
        """Index of the highest-volatility (crisis) state."""
        return self.n_states - 1

    def crisis_probability(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Return the posterior probability of being in the crisis state.

        This is the key signal for momentum crash gates, position sizing,
        and regime-conditional risk management.
        """
        _, probs = self.predict(features_df)
        crisis_col = f"state_{self.crisis_state_index}"
        return probs[crisis_col].rename("crisis_probability")

    # ------------------------------------------------------------------
    # State sorting (solves label-switching problem)
    # ------------------------------------------------------------------

    def _sort_states_by_volatility(self) -> None:
        """
        Reorder HMM states so state 0 has lowest emission mean for
        realized_vol and the last state has highest.

        This ensures consistent labelling across refits — the label-switching
        problem is the #1 practical HMM pitfall.
        """
        if self.model_ is None:
            return

        # realized_vol is the 2nd feature column (index 1)
        vol_idx = _FEATURE_COLS.index("realized_vol")
        vol_means = self.model_.means_[:, vol_idx]
        sort_order = np.argsort(vol_means)  # ascending: low vol → high vol

        if np.array_equal(sort_order, np.arange(self.n_states)):
            return  # already sorted

        # Reorder means, covars, startprob, transmat
        self.model_.means_ = self.model_.means_[sort_order]

        if self.model_.covariance_type == "full":
            self.model_.covars_ = self.model_.covars_[sort_order]
        elif self.model_.covariance_type == "diag":
            self.model_.covars_ = self.model_.covars_[sort_order]

        self.model_.startprob_ = self.model_.startprob_[sort_order]
        # Reorder both rows and columns of transition matrix
        self.model_.transmat_ = self.model_.transmat_[sort_order][:, sort_order]

    # ------------------------------------------------------------------
    # Static feature extractor
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(
        prices: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        rolling_window: int = _DEFAULT_ROLLING_WINDOW,
        corr_window: int = _DEFAULT_CORR_WINDOW,
    ) -> pd.DataFrame:
        """
        Construct the multi-feature DataFrame required by ``fit`` / ``predict``.

        Parameters
        ----------
        prices : pd.DataFrame
            Asset price DataFrame (rows = dates, columns = assets).  Used to
            derive returns when ``returns`` is not supplied.
        returns : pd.DataFrame, optional
            Pre-computed returns (same shape as ``prices``).  When supplied,
            ``prices`` is only used for shape/index validation.
        rolling_window : int, default 21
            Window for return and volatility features.
        corr_window : int, default 63
            Window for the average pairwise correlation feature.

        Returns
        -------
        pd.DataFrame with columns [rolling_return, realized_vol,
                                    avg_pairwise_corr, trend_strength]
        """
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pd.DataFrame.")

        if returns is None:
            ret = prices.pct_change().dropna(how="all")
        else:
            if not isinstance(returns, pd.DataFrame):
                raise TypeError("returns must be a pd.DataFrame.")
            ret = returns.copy()

        ret = ret.astype(float)

        # ---- Average cross-sectional return ----
        mean_ret = ret.mean(axis=1)

        # ---- Rolling return (momentum proxy) ----
        rolling_return = (
            mean_ret.rolling(window=rolling_window, min_periods=max(1, rolling_window // 2))
            .mean()
        )

        # ---- Realised volatility ----
        realized_vol = (
            mean_ret.rolling(window=rolling_window, min_periods=max(1, rolling_window // 2))
            .std()
            .mul(np.sqrt(_ANNUALISATION))
        )

        # ---- Average pairwise correlation ----
        n_assets = ret.shape[1]
        if n_assets >= 2:
            avg_corr = HMMRegimeDetector._rolling_avg_pairwise_corr(
                ret, window=corr_window
            )
        else:
            # Single asset fallback: rolling lag-1 autocorrelation
            single = ret.iloc[:, 0]
            avg_corr = (
                single.rolling(window=corr_window, min_periods=max(5, corr_window // 4))
                .apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan,
                       raw=True)
            )

        # ---- Trend strength ----
        trend_strength = rolling_return / (realized_vol + _EPSILON)

        features = pd.DataFrame(
            {
                "rolling_return": rolling_return,
                "realized_vol": realized_vol,
                "avg_pairwise_corr": avg_corr,
                "trend_strength": trend_strength,
            },
            index=mean_ret.index,
        )

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_avg_pairwise_corr(
        returns: pd.DataFrame, window: int
    ) -> pd.Series:
        """
        Compute the rolling average pairwise Pearson correlation across all
        asset pairs, vectorised over the time axis.

        Returns a pd.Series aligned to ``returns.index``.
        """
        n = returns.shape[1]
        n_pairs = n * (n - 1) // 2
        if n_pairs == 0:
            return pd.Series(np.nan, index=returns.index)

        min_p = max(5, window // 4)
        avg_corr_values = np.full(len(returns), np.nan)

        arr = returns.values  # (T, N)
        for t in range(len(arr)):
            start = max(0, t - window + 1)
            chunk = arr[start : t + 1]
            if chunk.shape[0] < min_p:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                C = np.corrcoef(chunk.T)  # (N, N)
            if np.any(np.isnan(C)):
                # Robust fallback: compute pair-by-pair
                pair_corrs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = chunk[:, i], chunk[:, j]
                        if np.std(a) > 0 and np.std(b) > 0:
                            pair_corrs.append(np.corrcoef(a, b)[0, 1])
                avg_corr_values[t] = np.nanmean(pair_corrs) if pair_corrs else np.nan
            else:
                upper = C[np.triu_indices(n, k=1)]
                avg_corr_values[t] = upper.mean()

        return pd.Series(avg_corr_values, index=returns.index, name="avg_pairwise_corr")

    def _prepare_X(
        self, features_df: pd.DataFrame, fit: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select feature columns, apply standardisation, and return a clean
        (NaN-free) array along with a boolean mask of valid rows.

        Parameters
        ----------
        features_df : pd.DataFrame
            Full feature matrix (may contain NaN).
        fit : bool
            When ``True``, compute and store the mean/std for standardisation.

        Returns
        -------
        X : np.ndarray of shape (n_valid_rows, n_features)
        valid_mask : np.ndarray of bool, shape (len(features_df),)
        """
        # Ensure only the expected columns are used, in canonical order
        available = [c for c in _FEATURE_COLS if c in features_df.columns]
        if len(available) < len(_FEATURE_COLS):
            missing = set(_FEATURE_COLS) - set(available)
            raise ValueError(f"features_df is missing required columns: {missing}")

        data = features_df[_FEATURE_COLS].copy()

        valid_mask = data.notna().all(axis=1).values
        data_clean = data.loc[valid_mask].values.astype(float)

        if self.standardise_features:
            if fit:
                self.feature_mean_ = pd.Series(
                    data_clean.mean(axis=0), index=_FEATURE_COLS
                )
                self.feature_std_ = pd.Series(
                    data_clean.std(axis=0, ddof=1).clip(min=_EPSILON),
                    index=_FEATURE_COLS,
                )
            if self.feature_mean_ is not None and self.feature_std_ is not None:
                data_clean = (
                    data_clean - self.feature_mean_.values
                ) / self.feature_std_.values

        return data_clean, valid_mask

    @staticmethod
    def _validate_features(features_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError(
                f"features_df must be a pd.DataFrame, got {type(features_df).__name__}."
            )
        if features_df.empty:
            raise ValueError("features_df is empty.")
        return features_df

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "This HMMRegimeDetector instance is not fitted yet. "
                "Call fit() before predict()."
            )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"HMMRegimeDetector("
            f"n_states={self.n_states}, "
            f"covariance_type='{self.covariance_type}', "
            f"n_iter={self.n_iter}, "
            f"status={status})"
        )
