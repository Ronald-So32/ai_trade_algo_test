"""
ML Alpha Combination Strategy
===============================
Combines multiple weak alpha signals via gradient-boosted trees to predict
cross-sectional stock returns, then constructs a long-short portfolio from
the predictions.

Research Basis
--------------
- Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning":
  GBTs dominate for cross-sectional return prediction. Rank-transformed
  features and nonlinear interactions are the main sources of gain.
  Realistic OOS IC: 0.02-0.05. Top features: momentum, liquidity, vol.
- López de Prado (2018) "Advances in Financial Machine Learning":
  Purged walk-forward CV with embargo gap prevents lookahead.
  Fractional Kelly (quarter to half) for position sizing.
  Rolling windows preferred over expanding for regime-shifting markets.
- Kakushadze (2016) "101 Formulaic Alphas": Signal combination via
  rank-weighted composites outperforms individual signals.
- DeMiguel et al. (2020): Combining many weak signals with regularisation
  beats concentrated bets on single strong signals.

Implementation
--------------
1. Feature construction: 15+ alpha signals at multiple horizons (vol regime,
   vov, recovery, turnover, mean-reversion speed, momentum interactions)
   with cross-sectional rank transformation (GKX finding).
2. Rolling-window walk-forward training with purge gap + embargo (LdP).
3. GBT predicts 5-day forward cross-sectional return rank (less noisy
   than 1-day, better matches signal half-lives).
4. Long top quintile, short bottom quintile, blended equal-weight +
   fractional-Kelly confidence weighting.
5. Turnover penalty: exponential smoothing of weights to reduce churn.
6. Drawdown-aware continuous position sizing (CDaR-inspired).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMRegressor as _GBTRegressor
    _USE_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as _GBTRegressor
    _USE_LGBM = False


class MLAlphaStrategy(Strategy):
    """
    ML-driven cross-sectional equity strategy combining discovered alpha
    signals via gradient-boosted trees with walk-forward training.

    Parameters
    ----------
    n_estimators : int
        Boosting rounds (default 300). More trees with low LR = better
        generalisation per Friedman (2001).
    max_depth : int
        Maximum tree depth (default 3). Shallow trees reduce overfitting
        on noisy financial data.
    learning_rate : float
        Boosting learning rate (default 0.03).
    min_child_samples : int
        Min observations per leaf (default 100). High value is critical
        for financial data per GKX.
    subsample : float
        Row subsampling (default 0.6). Stochastic gradient boosting.
    colsample_bytree : float
        Feature subsampling (default 0.6). Forces model to use diverse
        features rather than overfitting to dominant ones.
    retrain_every : int
        Retrain every N days (default 63 = quarterly).
    train_window : int
        Rolling training window in days (default 1260 = 5yr). Rolling
        preferred over expanding for regime adaptation (LdP).
    purge_gap : int
        Days between train end and prediction (default 10). Includes
        embargo for the 5-day forward return target.
    fwd_horizon : int
        Forward return horizon in days (default 5). Reduces noise vs
        1-day, better matches signal half-lives.
    quintile : float
        Fraction of universe per leg (default 0.20).
    target_gross : float
        Target gross exposure (default 1.0).
    kelly_fraction : float
        Fractional Kelly (default 0.20). Quarter-Kelly is conservative
        but robust to estimation error (LdP recommendation).
    weight_smoothing : float
        Exponential smoothing factor for weights (default 0.3). Reduces
        turnover: w_t = alpha * w_new + (1-alpha) * w_{t-1}.
    max_dd_threshold : float
        Drawdown threshold for exposure scaling (default 0.15).
    dd_scale_floor : float
        Minimum exposure during drawdown (default 0.20).
    """

    RESEARCH_GROUNDING = {
        "academic_basis": (
            "Gu, Kelly & Xiu (2020) empirical asset pricing via ML; "
            "López de Prado (2018) purged walk-forward CV; "
            "DeMiguel et al. (2020) signal combination with regularisation; "
            "Kakushadze (2016) 101 formulaic alphas"
        ),
        "historical_evidence": (
            "GBT combinations of rank-transformed technical signals achieve "
            "IC of 0.02-0.05 cross-sectionally. Top predictors across all "
            "models: momentum variants, liquidity, and volatility features."
        ),
        "implementation_risks": (
            "Overfitting to historical patterns; regime changes degrading "
            "learned relationships; signal decay (5.6% annually in US); "
            "model instability near retraining boundaries; estimation error "
            "in Kelly sizing."
        ),
        "realistic_expectations": (
            "Adds diversification value to multi-strategy portfolio. "
            "Expect modest standalone Sharpe but meaningful MaxDD reduction "
            "through low correlation with momentum/value strategies."
        ),
    }

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        min_child_samples: int = 100,
        subsample: float = 0.6,
        colsample_bytree: float = 0.6,
        retrain_every: int = 63,
        train_window: int = 1260,
        purge_gap: int = 10,
        fwd_horizon: int = 5,
        quintile: float = 0.20,
        target_gross: float = 1.0,
        kelly_fraction: float = 0.20,
        weight_smoothing: float = 0.3,
        max_dd_threshold: float = 0.15,
        dd_scale_floor: float = 0.20,
    ) -> None:
        params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            retrain_every=retrain_every,
            train_window=train_window,
            purge_gap=purge_gap,
            fwd_horizon=fwd_horizon,
            quintile=quintile,
            target_gross=target_gross,
            kelly_fraction=kelly_fraction,
            weight_smoothing=weight_smoothing,
            max_dd_threshold=max_dd_threshold,
            dd_scale_floor=dd_scale_floor,
        )
        super().__init__(name="MLAlphaStrategy", params=params)
        self._model = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_features(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Construct per-stock, per-date feature matrices.

        Multi-horizon design (GKX finding): each signal family is computed
        at short/medium/long windows to capture different frequency effects.
        All features are cross-sectionally rank-transformed to [0, 1].
        """
        features = {}

        # --- Volatility features (3 horizons) ---
        for window, label in [(10, "10d"), (21, "21d"), (63, "63d")]:
            mp = max(5, window // 2)
            vol = returns.rolling(window, min_periods=mp).std() * np.sqrt(252)
            features[f"vol_{label}"] = vol

        # --- Volatility-of-volatility (uncertainty) ---
        vol_21 = returns.rolling(21, min_periods=10).std() * np.sqrt(252)
        vol_63 = returns.rolling(63, min_periods=30).std() * np.sqrt(252)
        features["vov_21d"] = vol_21.rolling(21, min_periods=10).std()
        features["vov_63d"] = vol_63.rolling(63, min_periods=30).std()

        # --- Vol term structure (short/long vol ratio) ---
        vol_5 = returns.rolling(5, min_periods=3).std() * np.sqrt(252)
        features["vol_ts_5v21"] = np.log(
            vol_5.clip(lower=1e-8)
        ) - np.log(vol_21.clip(lower=1e-8))
        features["vol_ts_21v63"] = np.log(
            vol_21.clip(lower=1e-8)
        ) - np.log(vol_63.clip(lower=1e-8))

        # --- Recovery / distance features (2 horizons) ---
        for window, label in [(63, "63d"), (252, "252d")]:
            mp = window // 2
            roll_min = prices.rolling(window, min_periods=mp).min()
            roll_max = prices.rolling(window, min_periods=mp).max()
            features[f"dist_low_{label}"] = prices / roll_min - 1
            features[f"dist_high_{label}"] = prices / roll_max - 1

        # --- Mean-reversion speed (OU process, 2 horizons) ---
        for window, label in [(63, "63d"), (126, "126d")]:
            mp = window // 2
            ar1 = returns.rolling(window, min_periods=mp).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]
                if len(x) > 2 else 0,
                raw=True,
            )
            features[f"ou_speed_{label}"] = -np.log(ar1.abs().clip(lower=0.01))

        # --- Momentum features (3 horizons, risk-adjusted) ---
        for window, label in [(21, "21d"), (63, "63d"), (126, "126d")]:
            mp = window // 2
            ret_roll = returns.rolling(window, min_periods=mp).mean() * 252
            vol_roll = returns.rolling(window, min_periods=mp).std() * np.sqrt(252)
            vol_safe = vol_roll.replace(0, np.nan)
            features[f"risk_adj_mom_{label}"] = ret_roll / vol_safe

        # --- Momentum consistency (fraction of positive days) ---
        features["mom_consistency_63d"] = returns.rolling(63, min_periods=30).apply(
            lambda x: (x > 0).mean(), raw=True
        )

        # --- Return z-score (mean-reversion proxy, 2 horizons) ---
        for window, label in [(21, "21d"), (63, "63d")]:
            mp = window // 2
            ret_mean = returns.rolling(window, min_periods=mp).mean()
            ret_std = returns.rolling(window, min_periods=mp).std().replace(0, np.nan)
            features[f"ret_zscore_{label}"] = (returns - ret_mean) / ret_std

        # --- Liquidity / turnover features ---
        if volumes is not None:
            log_vol = np.log(volumes.clip(lower=1))
            vol_s = log_vol.rolling(21, min_periods=10).mean()
            vol_l = log_vol.rolling(126, min_periods=60).mean()
            features["turnover_chg_21d"] = vol_s - vol_l
            features["vol_trend_5d"] = (
                log_vol.rolling(5, min_periods=3).mean() - vol_s
            )

        # --- Cross-sectional rank transform (GKX) ---
        ranked_features = {}
        for name, feat_df in features.items():
            ranked_features[name] = feat_df.rank(axis=1, pct=True)

        self._feature_names = sorted(ranked_features.keys())
        return ranked_features

    def _stack_features(
        self,
        features: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """Stack all features for a single date into (n_assets, n_features)."""
        first_feat = next(iter(features.values()))
        if date not in first_feat.index:
            return None

        rows = {}
        for name in self._feature_names:
            if date in features[name].index:
                rows[name] = features[name].loc[date]

        if not rows:
            return None

        return pd.DataFrame(rows).dropna()

    def _stack_features_panel(
        self,
        features: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
        returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Stack features and forward returns into training matrix.
        Target: cross-sectional rank of N-day forward return.
        """
        fwd_horizon = self.params["fwd_horizon"]
        X_parts = []
        y_parts = []

        # N-day forward return (less noisy than 1-day)
        fwd_ret = returns.rolling(fwd_horizon).sum().shift(-fwd_horizon)

        for date in dates:
            feat_df = self._stack_features(features, date)
            if feat_df is None or len(feat_df) < 10:
                continue
            if date not in fwd_ret.index:
                continue

            fwd = fwd_ret.loc[date].reindex(feat_df.index).dropna()
            common = feat_df.index.intersection(fwd.index)
            if len(common) < 10:
                continue

            feat_sub = feat_df.loc[common]
            y_ranked = fwd.loc[common].rank(pct=True)

            X_parts.append(feat_sub)
            y_parts.append(y_ranked)

        if not X_parts:
            return pd.DataFrame(), pd.Series(dtype=float)

        return pd.concat(X_parts, axis=0), pd.concat(y_parts, axis=0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _build_model(self):
        """Instantiate a fresh GBT with configured hyperparameters."""
        p = self.params
        if _USE_LGBM:
            return _GBTRegressor(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                learning_rate=p["learning_rate"],
                min_child_samples=p["min_child_samples"],
                subsample=p["subsample"],
                colsample_bytree=p["colsample_bytree"],
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            return _GBTRegressor(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                learning_rate=p["learning_rate"],
                min_samples_leaf=p["min_child_samples"],
                subsample=p["subsample"],
                random_state=42,
            )

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Walk-forward ML signal generation with rolling window training.

        For each prediction date:
        1. Train GBT on rolling window of (train_window) days with purge gap.
        2. Predict cross-sectional return rank for all assets.
        3. Long top quintile, short bottom quintile.
        4. Blend equal-weight with fractional-Kelly confidence weighting.
        5. Exponentially smooth weights to reduce turnover.
        """
        volumes = kwargs.get("volumes", None)
        train_window = self.params["train_window"]
        retrain_every = self.params["retrain_every"]
        purge_gap = self.params["purge_gap"]
        quintile = self.params["quintile"]
        kelly_f = self.params["kelly_fraction"]
        smoothing = self.params["weight_smoothing"]

        # Need train_window + purge_gap before first prediction
        min_start = train_window + purge_gap

        logger.info("MLAlphaStrategy: building features (%d assets)...", prices.shape[1])
        features = self._build_features(prices, returns, volumes)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        all_dates = prices.index
        model = None
        last_train_idx = -retrain_every
        prev_signals = None  # for smoothing

        logger.info(
            "MLAlphaStrategy: walk-forward over %d dates "
            "(window=%d, retrain=%dd, purge=%dd, fwd=%dd)...",
            len(all_dates), train_window, retrain_every, purge_gap,
            self.params["fwd_horizon"],
        )

        for i, date in enumerate(all_dates):
            if i < min_start:
                continue

            # Retrain on rolling window
            if model is None or (i - last_train_idx) >= retrain_every:
                train_end = i - purge_gap
                train_start = max(0, train_end - train_window)

                if train_end - train_start < 252:
                    continue

                train_dates = all_dates[train_start:train_end]
                X_train, y_train = self._stack_features_panel(
                    features, train_dates, returns
                )

                if len(X_train) < 500:
                    continue

                model = self._build_model()
                model.fit(X_train.values, y_train.values)
                last_train_idx = i

                logger.debug(
                    "  Retrained at %s: %d samples, %d features, window [%s, %s]",
                    date.date(), len(X_train), X_train.shape[1],
                    all_dates[train_start].date(), all_dates[train_end - 1].date(),
                )

            if model is None:
                continue

            # Predict
            feat_today = self._stack_features(features, date)
            if feat_today is None or len(feat_today) < 5:
                continue

            pred_scores = model.predict(feat_today.values)
            pred_series = pd.Series(pred_scores, index=feat_today.index)

            # Quintile long/short
            n_assets = len(pred_series)
            n_select = max(1, int(np.floor(n_assets * quintile)))
            ranked = pred_series.rank(ascending=True)
            long_mask = ranked >= (ranked.max() - n_select + 1)
            short_mask = ranked <= n_select
            long_assets = long_mask[long_mask].index
            short_assets = short_mask[short_mask].index

            # Fractional Kelly: blend equal-weight with confidence-weighted
            median_pred = pred_series.median()
            day_signals = pd.Series(0.0, index=prices.columns)

            if len(long_assets) > 0:
                confidence = (pred_series[long_assets] - median_pred).clip(lower=0)
                conf_sum = confidence.sum()
                if conf_sum > 0:
                    eq_w = 1.0 / len(long_assets)
                    conf_w = confidence / conf_sum
                    blended = (1 - kelly_f) * eq_w + kelly_f * conf_w
                    day_signals[long_assets] = blended / blended.sum()
                else:
                    day_signals[long_assets] = 1.0 / len(long_assets)

            if len(short_assets) > 0:
                confidence = (median_pred - pred_series[short_assets]).clip(lower=0)
                conf_sum = confidence.sum()
                if conf_sum > 0:
                    eq_w = 1.0 / len(short_assets)
                    conf_w = confidence / conf_sum
                    blended = (1 - kelly_f) * eq_w + kelly_f * conf_w
                    day_signals[short_assets] = -(blended / blended.sum())
                else:
                    day_signals[short_assets] = -1.0 / len(short_assets)

            # Exponential smoothing to reduce turnover
            if prev_signals is not None:
                day_signals = smoothing * day_signals + (1 - smoothing) * prev_signals
                # Re-normalise long/short legs after smoothing
                long_total = day_signals.clip(lower=0).sum()
                short_total = day_signals.clip(upper=0).abs().sum()
                if long_total > 0:
                    day_signals[day_signals > 0] /= long_total
                if short_total > 0:
                    day_signals[day_signals < 0] /= short_total
                    day_signals[day_signals < 0] *= -1
                    day_signals[day_signals < 0] *= -1

            prev_signals = day_signals.copy()
            signals.loc[date] = day_signals

        logger.info("MLAlphaStrategy: signal generation complete.")
        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convert signals to portfolio weights with:
        1. Target gross exposure scaling
        2. Drawdown-aware continuous position sizing (CDaR-inspired)
        """
        if returns is None:
            returns = kwargs.get("returns", None)

        target_gross = self.params["target_gross"]
        max_dd_threshold = self.params["max_dd_threshold"]
        dd_floor = self.params["dd_scale_floor"]

        # Scale to target gross exposure
        long_sum = signals.clip(lower=0).sum(axis=1).replace(0, np.nan)
        short_sum = signals.clip(upper=0).abs().sum(axis=1).replace(0, np.nan)

        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for col in signals.columns:
            is_long = signals[col] > 0
            is_short = signals[col] < 0
            weights.loc[is_long, col] = (
                signals.loc[is_long, col] / long_sum[is_long] * (target_gross / 2.0)
            )
            weights.loc[is_short, col] = (
                signals.loc[is_short, col] / short_sum[is_short] * (target_gross / 2.0)
            )
        weights = weights.fillna(0.0)

        # Continuous drawdown scaling
        if returns is not None:
            strat_returns = (weights.shift(1) * returns).sum(axis=1)
            cum = (1 + strat_returns).cumprod()
            peak = cum.cummax()
            dd = (cum - peak) / peak

            soft_start = max_dd_threshold * 0.5
            hard_limit = max_dd_threshold * 0.95

            dd_abs = dd.abs()
            scale = pd.Series(1.0, index=weights.index)
            mask_mid = (dd_abs > soft_start) & (dd_abs < hard_limit)
            mask_hard = dd_abs >= hard_limit

            frac = (dd_abs[mask_mid] - soft_start) / (hard_limit - soft_start)
            scale[mask_mid] = 1.0 - frac * (1.0 - dd_floor)
            scale[mask_hard] = dd_floor

            weights = weights.mul(scale, axis=0)

        return weights
