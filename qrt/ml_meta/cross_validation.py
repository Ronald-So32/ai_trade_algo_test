"""
Time-Series Cross-Validation
=============================
Purged time-series cross-validation that wraps sklearn's
``TimeSeriesSplit`` and enforces a gap between each training fold and its
corresponding test fold to prevent information leakage.

Classes
-------
TimeSeriesCV
    Purged cross-validator with configurable gap and configurable number
    of splits.  Supports arbitrary sklearn-compatible estimators via
    ``cross_validate``.
"""

from __future__ import annotations

import logging
from typing import Any, Generator, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
)

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """
    Purged time-series cross-validator.

    Wraps :class:`sklearn.model_selection.TimeSeriesSplit` and removes
    ``gap_days`` observations from the end of every training fold to
    prevent leakage between training and test sets (i.e. the model cannot
    inadvertently see information derived from the test window during
    training).

    Parameters
    ----------
    n_splits : int, default 5
        Number of cross-validation folds.
    max_train_size : int or None, default None
        If set, cap each training fold at this many observations (rolling
        window cross-validation).
    test_size : int or None, default None
        Fixed size for each test fold.  When ``None`` sklearn determines
        the test size automatically.

    Examples
    --------
    >>> cv = TimeSeriesCV(n_splits=5)
    >>> for train_idx, test_idx in cv.split(X, gap_days=5):
    ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
    ...     preds = model.predict(X.iloc[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size

        self._tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        gap_days: int = 5,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged (train, test) index pairs.

        A gap of ``gap_days`` rows is removed from the *end* of each
        training fold so that any information that could leak across
        ``gap_days`` periods of look-back is excluded.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.  Only the length is used for splitting.
        gap_days : int, default 5
            Number of rows to strip from the tail of the training fold.
            Set to 0 for no purging (equivalent to plain TimeSeriesSplit).

        Yields
        ------
        train_idx : np.ndarray
            Integer positions of training observations.
        test_idx : np.ndarray
            Integer positions of test observations.
        """
        if gap_days < 0:
            raise ValueError(f"gap_days must be >= 0, got {gap_days}.")

        n = len(X)
        for raw_train, test_idx in self._tscv.split(X):
            # Purge: drop the last `gap_days` rows of the training fold
            if gap_days > 0:
                train_idx = raw_train[: max(0, len(raw_train) - gap_days)]
            else:
                train_idx = raw_train

            # Skip degenerate folds that become empty after purging
            if len(train_idx) == 0:
                logger.warning(
                    "Skipping fold: training set is empty after purging "
                    "%d gap days from %d raw training observations.",
                    gap_days,
                    len(raw_train),
                )
                continue

            logger.debug(
                "Fold — train: [%d, %d] (%d obs), test: [%d, %d] (%d obs), gap: %d",
                train_idx[0],
                train_idx[-1],
                len(train_idx),
                test_idx[0],
                test_idx[-1],
                len(test_idx),
                gap_days,
            )
            yield train_idx, test_idx

    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        gap_days: int = 5,
        scoring: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """
        Evaluate *model* using purged time-series cross-validation.

        For classification targets the method computes accuracy and
        ROC-AUC per fold (where AUC is available).  For regression
        targets it computes root-mean-squared error and mean-squared
        error.  Target type is inferred from ``y``.

        Parameters
        ----------
        model : sklearn-compatible estimator
            Must implement ``fit(X, y)`` and ``predict(X)``.
            Optionally implements ``predict_proba(X)`` for AUC scoring.
        X : pd.DataFrame
            Feature matrix aligned with ``y``.
        y : pd.Series or pd.DataFrame
            Target variable(s).
        gap_days : int, default 5
            Passed through to :meth:`split`.
        scoring : list[str] or None
            Reserved for future use; currently ignored.

        Returns
        -------
        dict
            Keys are metric names; values are lists of per-fold scores.
            Always contains ``"n_train"`` and ``"n_test"`` (fold sizes).

            Classification keys: ``"accuracy"``, ``"auc"`` (when
            ``predict_proba`` is available).

            Regression keys: ``"rmse"``, ``"mse"``.
        """
        results: dict[str, list[float]] = {
            "n_train": [],
            "n_test": [],
        }

        is_classification = self._is_classification(y)
        if is_classification:
            results["accuracy"] = []
            results["auc"] = []
        else:
            results["rmse"] = []
            results["mse"] = []

        for fold_num, (train_idx, test_idx) in enumerate(
            self.split(X, gap_days=gap_days), start=1
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            if isinstance(y, pd.DataFrame):
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
            else:
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            results["n_train"].append(len(train_idx))
            results["n_test"].append(len(test_idx))

            if is_classification:
                acc = accuracy_score(y_test, preds)
                results["accuracy"].append(float(acc))

                # AUC — requires predict_proba and binary / multi-class target
                auc = self._try_compute_auc(model, X_test, y_test)
                results["auc"].append(auc)

                logger.debug(
                    "Fold %d — accuracy: %.4f, AUC: %.4f  "
                    "(train=%d, test=%d)",
                    fold_num,
                    acc,
                    auc,
                    len(train_idx),
                    len(test_idx),
                )
            else:
                mse = float(mean_squared_error(y_test, preds))
                rmse = float(np.sqrt(mse))
                results["mse"].append(mse)
                results["rmse"].append(rmse)

                logger.debug(
                    "Fold %d — RMSE: %.6f  (train=%d, test=%d)",
                    fold_num,
                    rmse,
                    len(train_idx),
                    len(test_idx),
                )

        if not results["n_train"]:
            logger.warning("cross_validate produced zero valid folds.")

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_effective_splits(self) -> int:
        """Nominal number of splits (actual folds may be fewer after purging)."""
        return self.n_splits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_classification(y: pd.Series | pd.DataFrame) -> bool:
        """
        Heuristic: treat integer or boolean dtype as classification.
        Float dtype is treated as regression.
        """
        if isinstance(y, pd.DataFrame):
            dtypes = y.dtypes
            return all(
                pd.api.types.is_integer_dtype(dt) or pd.api.types.is_bool_dtype(dt)
                for dt in dtypes
            )
        return pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y)

    @staticmethod
    def _try_compute_auc(
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
        """
        Attempt to compute ROC-AUC; return ``float('nan')`` on failure.
        """
        if not hasattr(model, "predict_proba"):
            return float("nan")
        try:
            proba = model.predict_proba(X_test)
            n_classes = proba.shape[1] if proba.ndim == 2 else 1
            if n_classes == 2:
                return float(roc_auc_score(y_test, proba[:, 1]))
            # Multi-class OVR macro-average
            return float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("AUC computation failed: %s", exc)
            return float("nan")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TimeSeriesCV("
            f"n_splits={self.n_splits}, "
            f"max_train_size={self.max_train_size}, "
            f"test_size={self.test_size})"
        )
