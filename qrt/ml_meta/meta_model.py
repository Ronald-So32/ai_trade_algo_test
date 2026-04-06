"""
ML Meta-Model
==============
Ensemble of machine learning models that predict per-strategy
performance and dynamically adjust strategy weights.

Three prediction tasks are solved jointly:

1. **Expected return** — a gradient-boosted regressor forecasts each
   strategy's next-period return.
2. **Outperformance probability** — a logistic-regression classifier and
   a random-forest classifier each estimate the probability that a
   strategy's next-period return exceeds the cross-sectional median.
3. **Weight adjustment** — the outperformance probabilities and expected
   return ranks are combined to produce a multiplicative weight
   adjustment factor that the caller can apply to any base-weight vector.

Classes
-------
MetaModel
    Stateful ensemble; call :meth:`fit` once on training data, then
    :meth:`predict` on new feature matrices.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from qrt.ml_meta.cross_validation import TimeSeriesCV

logger = logging.getLogger(__name__)

# Supported model type identifiers
_SUPPORTED_MODELS: set[str] = {
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
}

# Default hyper-parameters (conservative, production-safe defaults)
_DEFAULT_LR_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "C": 1.0,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": 42,
}
_DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}
_DEFAULT_GB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 20,
    "random_state": 42,
}


class MetaModel:
    """
    ML meta-model that predicts strategy performance and adjusts weights.

    The meta-model trains a collection of classifiers and a regressor
    (one per strategy) and combines their predictions into actionable
    weight-adjustment signals.

    Parameters
    ----------
    models : list[str] or None
        Model types to include.  Supported values:
        ``"logistic_regression"``, ``"random_forest"``,
        ``"gradient_boosting"``.  Defaults to all three.
    n_cv_splits : int, default 5
        Number of folds for :class:`~qrt.ml_meta.cross_validation.TimeSeriesCV`
        used in :meth:`fit`.
    cv_gap_days : int, default 5
        Purge gap (days) between training and test folds during CV.
    weight_adj_clip : float, default 2.0
        Maximum multiplicative weight-adjustment factor (symmetric
        around 1.0).  Prevents a single model from dominating.
    min_weight : float, default 0.0
        Floor applied to any adjusted weight before re-normalisation.

    Attributes
    ----------
    classifiers_ : dict[str, dict[str, Any]]
        Nested dict ``{strategy_name: {model_type: fitted_estimator}}``.
    regressors_ : dict[str, Any]
        Dict ``{strategy_name: fitted_regressor}`` for expected-return
        prediction.
    strategy_names_ : list[str] or None
        Strategy names seen during :meth:`fit`.
    is_fitted_ : bool
        Whether the model has been fitted.
    cv_scores_ : dict
        Cross-validation scores from the last call to :meth:`fit`.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        n_cv_splits: int = 5,
        cv_gap_days: int = 5,
        weight_adj_clip: float = 2.0,
        min_weight: float = 0.0,
    ) -> None:
        if models is None:
            models = ["logistic_regression", "random_forest", "gradient_boosting"]

        unknown = set(models) - _SUPPORTED_MODELS
        if unknown:
            raise ValueError(
                f"Unsupported model type(s): {unknown}. "
                f"Choose from {_SUPPORTED_MODELS}."
            )
        if not models:
            raise ValueError("models list must not be empty.")

        self.models = list(models)
        self.n_cv_splits = n_cv_splits
        self.cv_gap_days = cv_gap_days
        self.weight_adj_clip = weight_adj_clip
        self.min_weight = min_weight

        # Fitted state
        self.classifiers_: dict[str, dict[str, Any]] = {}
        self.regressors_: dict[str, Any] = {}
        self.strategy_names_: list[str] | None = None
        self.is_fitted_: bool = False
        self.cv_scores_: dict = {}

    # ------------------------------------------------------------------
    # Feature and target construction
    # ------------------------------------------------------------------

    def build_features(
        self,
        strategy_signals: dict[str, pd.Series],
        volatility: pd.Series,
        correlation: pd.Series,
        drawdown: pd.Series,
        regime_probs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assemble a raw feature matrix from the provided market inputs.

        All inputs must share the same DatetimeIndex.  Columns in the
        returned DataFrame follow naming conventions expected by
        :class:`~qrt.ml_meta.feature_engineering.MetaFeatureEngineer`:

        * Signal columns are prefixed ``signal_<strategy_name>``.
        * Scalar columns are named ``volatility``, ``correlation``,
          ``drawdown``.
        * Regime-probability columns are prefixed ``regime_``.

        Parameters
        ----------
        strategy_signals : dict[str, pd.Series]
            Mapping from strategy name to its signal series.
        volatility : pd.Series
            Realised volatility (annualised) series.
        correlation : pd.Series
            Average pairwise cross-strategy correlation series.
        drawdown : pd.Series
            Current portfolio drawdown depth (non-positive values).
        regime_probs : pd.DataFrame
            Columns are regime probability columns (e.g.
            ``prob_low_vol``, …).

        Returns
        -------
        pd.DataFrame
            Raw feature matrix indexed by date.
        """
        frames: list[pd.DataFrame | pd.Series] = []

        for name, sig in strategy_signals.items():
            frames.append(sig.rename(f"signal_{name}"))

        frames.append(volatility.rename("volatility"))
        frames.append(correlation.rename("correlation"))
        frames.append(drawdown.rename("drawdown"))

        # Prefix regime probability columns to avoid name clashes
        renamed_regimes = regime_probs.add_prefix("regime_")
        frames.append(renamed_regimes)

        feature_df = pd.concat(frames, axis=1)

        logger.debug(
            "build_features — assembled %d features for %d dates.",
            feature_df.shape[1],
            feature_df.shape[0],
        )
        return feature_df

    def build_targets(
        self,
        strategy_returns: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Construct target variables from next-period strategy returns.

        For each strategy two targets are created:

        * ``<strategy>_return``      — the raw next-period return (regression target).
        * ``<strategy>_outperform``  — 1 if the return exceeds the
          cross-sectional median of all strategy returns on that date,
          0 otherwise (classification target).

        **Important**: Targets use a +1 shift on the *features* side rather
        than a -1 shift on returns.  Concretely, the target at row ``t`` is
        the return already realised at ``t`` — the caller is responsible for
        aligning features from ``t-1`` with targets at ``t``.  This avoids
        embedding a forward-looking shift(-1) into the target frame, which
        would leak future information if the caller naively joins features
        and targets on the same index.

        Parameters
        ----------
        strategy_returns : dict[str, pd.Series]
            Mapping from strategy name to its historical daily return series.

        Returns
        -------
        pd.DataFrame
            Target DataFrame with a DatetimeIndex matching the input series.
        """
        ret_frame = pd.DataFrame(strategy_returns).rename(
            columns={k: f"{k}_return" for k in strategy_returns}
        )

        # No forward-shift: targets are contemporaneous returns.
        # The caller must align features(t-1) → target(t) by shifting
        # the feature matrix forward by 1, NOT by shifting targets backward.
        # This prevents accidental lookahead if features and targets are
        # joined on the same date index without the caller remembering to
        # re-shift.

        # Cross-sectional median outperformance indicator
        raw_returns = pd.DataFrame(strategy_returns)
        median_ret = raw_returns.median(axis=1)
        for name in strategy_returns:
            ret_frame[f"{name}_outperform"] = (
                raw_returns[name].gt(median_ret).astype(int)
            )

        logger.debug(
            "build_targets — constructed %d target columns for %d dates.",
            ret_frame.shape[1],
            ret_frame.shape[0],
        )
        return ret_frame

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> "MetaModel":
        """
        Fit all models for each strategy using purged time-series CV.

        CV scores are stored in :attr:`cv_scores_` for inspection.
        The final fitted models are trained on the *full* dataset (not
        just the last fold) so they make use of all available history.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix produced by
            :meth:`build_features` (possibly after
            :class:`~qrt.ml_meta.feature_engineering.MetaFeatureEngineer`
            transformation).
        targets : pd.DataFrame
            Target matrix produced by :meth:`build_targets`.

        Returns
        -------
        self
        """
        # Align and drop rows with NaN in features or targets
        combined = features.join(targets, how="inner").dropna()
        if combined.empty:
            raise ValueError(
                "No complete rows remain after aligning features and targets "
                "and dropping NaNs."
            )

        feat_cols = list(features.columns)
        X = combined[feat_cols]

        # Identify strategy names from target columns
        return_cols = [c for c in targets.columns if c.endswith("_return")]
        strategies = [c.replace("_return", "") for c in return_cols]
        self.strategy_names_ = strategies

        cv = TimeSeriesCV(n_splits=self.n_cv_splits)
        self.cv_scores_ = {}
        self.classifiers_ = {}
        self.regressors_ = {}

        for strat in strategies:
            ret_col = f"{strat}_return"
            out_col = f"{strat}_outperform"

            if ret_col not in combined.columns or out_col not in combined.columns:
                logger.warning(
                    "Strategy '%s' is missing return or outperform columns; skipping.",
                    strat,
                )
                continue

            y_cls = combined[out_col].astype(int)
            y_reg = combined[ret_col].astype(float)

            logger.info(
                "Fitting meta-model for strategy '%s' on %d observations.",
                strat,
                len(X),
            )

            strat_scores: dict[str, dict] = {}

            # ---- Classification models --------------------------------
            self.classifiers_[strat] = {}
            for model_type in self.models:
                clf = self._build_classifier(model_type)

                # Cross-validate
                fold_scores = cv.cross_validate(
                    clf, X, y_cls, gap_days=self.cv_gap_days
                )
                strat_scores[model_type] = fold_scores

                # Final fit on full data
                clf.fit(X, y_cls)
                self.classifiers_[strat][model_type] = clf

                mean_acc = float(np.nanmean(fold_scores.get("accuracy", [np.nan])))
                mean_auc = float(np.nanmean(fold_scores.get("auc", [np.nan])))
                logger.info(
                    "  [%s / %s] CV accuracy=%.4f, AUC=%.4f",
                    strat,
                    model_type,
                    mean_acc,
                    mean_auc,
                )

            # ---- Gradient-boosting regressor (expected return) --------
            from sklearn.ensemble import GradientBoostingRegressor

            gb_reg = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )
            gb_reg.fit(X, y_reg)
            self.regressors_[strat] = gb_reg
            strat_scores["gb_regressor"] = {"fitted": True}

            self.cv_scores_[strat] = strat_scores

        self.is_fitted_ = True
        logger.info(
            "MetaModel fitted on %d strategies, %d observations.",
            len(strategies),
            len(X),
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Generate predictions from all fitted models.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix for the prediction period.

        Returns
        -------
        dict with keys:

        * ``"expected_return"``  — pd.DataFrame, one column per strategy,
          predicted next-period return from the GB regressor.
        * ``"outperform_prob"``  — pd.DataFrame, one column per strategy,
          ensemble average outperformance probability.
        * ``"weight_adjustment"`` — pd.DataFrame, one column per strategy,
          multiplicative weight-adjustment factors clipped to
          ``[1/clip, clip]``.
        """
        self._check_fitted()

        exp_ret: dict[str, pd.Series] = {}
        out_prob: dict[str, pd.Series] = {}

        for strat in self.strategy_names_:  # type: ignore[union-attr]
            # Expected return
            if strat in self.regressors_:
                exp_ret[strat] = pd.Series(
                    self.regressors_[strat].predict(features),
                    index=features.index,
                    name=strat,
                )
            else:
                exp_ret[strat] = pd.Series(
                    np.zeros(len(features)), index=features.index, name=strat
                )

            # Outperformance probability — ensemble average
            if strat in self.classifiers_:
                proba_list: list[np.ndarray] = []
                for model_type, clf in self.classifiers_[strat].items():
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(features)
                        # Column index 1 = positive class (outperforms)
                        if proba.shape[1] >= 2:
                            proba_list.append(proba[:, 1])
                if proba_list:
                    avg_proba = np.mean(proba_list, axis=0)
                else:
                    avg_proba = np.full(len(features), 0.5)
                out_prob[strat] = pd.Series(
                    avg_proba, index=features.index, name=strat
                )
            else:
                out_prob[strat] = pd.Series(
                    np.full(len(features), 0.5),
                    index=features.index,
                    name=strat,
                )

        expected_return_df = pd.DataFrame(exp_ret)
        outperform_prob_df = pd.DataFrame(out_prob)

        # Weight adjustment: linear mapping from outperform prob
        # [0, 1] -> [1/clip, clip] via 0.5 -> 1.0
        clip = self.weight_adj_clip
        weight_adj_df = outperform_prob_df.apply(
            lambda col: np.clip(2.0 * col * clip / clip, 1.0 / clip, clip),
            axis=0,
        )
        # More nuanced: adj = clip^(2*p-1) so p=0.5 -> 1, p=1 -> clip, p=0 -> 1/clip
        weight_adj_values = np.power(
            clip, 2.0 * outperform_prob_df.values - 1.0
        )
        weight_adj_df = pd.DataFrame(
            weight_adj_values,
            index=features.index,
            columns=outperform_prob_df.columns,
        )

        return {
            "expected_return": expected_return_df,
            "outperform_prob": outperform_prob_df,
            "weight_adjustment": weight_adj_df,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: dict[str, pd.DataFrame],
        actuals: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """
        Compute evaluation metrics for each strategy.

        Metrics computed per strategy:

        * ``accuracy``       — classification accuracy of outperformance
          prediction (threshold 0.5 on outperform_prob).
        * ``auc``            — ROC-AUC of outperformance probability.
        * ``ic``             — Pearson information coefficient between
          ``expected_return`` and actual return.
        * ``ic_rank``        — Spearman (rank) IC.
        * ``rmse``           — Root mean squared error of return forecast.

        Parameters
        ----------
        predictions : dict
            Output of :meth:`predict`.
        actuals : pd.DataFrame
            DataFrame with columns ``<strategy>_return`` and
            ``<strategy>_outperform`` aligned to the prediction index.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{strategy_name: {metric: value}}``.
        """
        self._check_fitted()

        out_prob_df = predictions.get("outperform_prob", pd.DataFrame())
        exp_ret_df = predictions.get("expected_return", pd.DataFrame())

        results: dict[str, dict[str, float]] = {}

        for strat in self.strategy_names_:  # type: ignore[union-attr]
            ret_col = f"{strat}_return"
            out_col = f"{strat}_outperform"

            if strat not in out_prob_df.columns:
                continue

            # Align index
            idx = out_prob_df.index.intersection(actuals.index)
            if idx.empty:
                logger.warning(
                    "No overlapping dates between predictions and actuals "
                    "for strategy '%s'.",
                    strat,
                )
                continue

            proba = out_prob_df.loc[idx, strat].values
            pred_labels = (proba >= 0.5).astype(int)
            metrics: dict[str, float] = {}

            if out_col in actuals.columns:
                true_labels = actuals.loc[idx, out_col].values.astype(int)
                metrics["accuracy"] = float(accuracy_score(true_labels, pred_labels))
                try:
                    metrics["auc"] = float(roc_auc_score(true_labels, proba))
                except Exception:
                    metrics["auc"] = float("nan")

            if ret_col in actuals.columns and strat in exp_ret_df.columns:
                actual_ret = actuals.loc[idx, ret_col].values
                forecast_ret = exp_ret_df.loc[idx, strat].values

                valid = ~(np.isnan(actual_ret) | np.isnan(forecast_ret))
                if valid.sum() > 1:
                    metrics["ic"] = float(
                        np.corrcoef(forecast_ret[valid], actual_ret[valid])[0, 1]
                    )
                    # Rank IC (Spearman)
                    from scipy.stats import spearmanr

                    rho, _ = spearmanr(forecast_ret[valid], actual_ret[valid])
                    metrics["ic_rank"] = float(rho)
                    mse = float(np.mean((forecast_ret[valid] - actual_ret[valid]) ** 2))
                    metrics["rmse"] = float(np.sqrt(mse))
                else:
                    metrics["ic"] = float("nan")
                    metrics["ic_rank"] = float("nan")
                    metrics["rmse"] = float("nan")

            results[strat] = metrics
            logger.info(
                "Evaluation [%s]: %s",
                strat,
                ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
            )

        return results

    # ------------------------------------------------------------------
    # Weight adjustment
    # ------------------------------------------------------------------

    def adjusted_weights(
        self,
        base_weights: dict[str, float],
        predictions: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """
        Adjust base strategy weights using meta-model predictions.

        The most recent row of ``predictions["weight_adjustment"]`` is
        multiplied element-wise with the base weights.  The result is
        clipped to ``[min_weight, inf)`` and re-normalised to sum to 1.

        Parameters
        ----------
        base_weights : dict[str, float]
            Current strategy weights (must be non-negative; need not
            sum to 1 before adjustment).
        predictions : dict
            Output of :meth:`predict`.

        Returns
        -------
        dict[str, float]
            Adjusted, normalised strategy weights.
        """
        self._check_fitted()

        adj_df = predictions.get("weight_adjustment", pd.DataFrame())
        if adj_df.empty:
            logger.warning(
                "weight_adjustment DataFrame is empty; returning base weights."
            )
            total = sum(base_weights.values())
            if total <= 0:
                n = len(base_weights)
                return {k: 1.0 / n for k in base_weights}
            return {k: v / total for k, v in base_weights.items()}

        # Use the latest available adjustment row
        latest_adj = adj_df.iloc[-1]

        adjusted: dict[str, float] = {}
        for strat, base_w in base_weights.items():
            adj_factor = float(latest_adj.get(strat, 1.0))
            new_w = max(base_w * adj_factor, self.min_weight)
            adjusted[strat] = new_w

        total = sum(adjusted.values())
        if total <= 0:
            logger.warning(
                "All adjusted weights are zero or negative; "
                "returning equal weights."
            )
            n = len(adjusted)
            return {k: 1.0 / n for k in adjusted}

        normalised = {k: v / total for k, v in adjusted.items()}

        logger.debug(
            "adjusted_weights — base: %s  ->  adjusted: %s",
            {k: f"{v:.4f}" for k, v in base_weights.items()},
            {k: f"{v:.4f}" for k, v in normalised.items()},
        )
        return normalised

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_classifier(self, model_type: str) -> Any:
        """Instantiate a fresh (unfitted) classifier by name."""
        if model_type == "logistic_regression":
            return LogisticRegression(**_DEFAULT_LR_PARAMS)
        if model_type == "random_forest":
            return RandomForestClassifier(**_DEFAULT_RF_PARAMS)
        if model_type == "gradient_boosting":
            return GradientBoostingClassifier(**_DEFAULT_GB_PARAMS)
        raise ValueError(f"Unknown model_type: '{model_type}'.")

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "MetaModel is not fitted. Call fit() before predict(), "
                "evaluate(), or adjusted_weights()."
            )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        strategies = (
            str(self.strategy_names_) if self.strategy_names_ else "unknown"
        )
        return (
            f"MetaModel("
            f"models={self.models}, "
            f"n_cv_splits={self.n_cv_splits}, "
            f"cv_gap_days={self.cv_gap_days}, "
            f"weight_adj_clip={self.weight_adj_clip}, "
            f"status={status}, "
            f"strategies={strategies})"
        )
