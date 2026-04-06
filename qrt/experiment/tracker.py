"""
ExperimentTracker: Records, stores, and compares quantitative research
experiments for full reproducibility.

Each experiment captures:
  - experiment_id   : UUID4 string
  - timestamp       : ISO-8601 UTC string
  - config_hash     : SHA-256 of the serialised config dict
  - dataset_version : SHA-256 of data file contents (if provided)
  - git_commit      : HEAD SHA from ``git rev-parse HEAD`` or "no-git"
  - strategy_set    : list of strategy names
  - parameters      : full config dict
  - results         : metrics dict (populated via log_result)
  - status          : "running" | "finished" | "failed"
  - duration_seconds: wall-clock time from start to finish
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _hash_dict(d: dict) -> str:
    """Return the SHA-256 hex digest of a JSON-serialised dict."""
    serialised = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def _hash_files(paths: list[str]) -> str:
    """
    Return a combined SHA-256 hex digest of the contents of *paths*.

    Files that do not exist are silently skipped; the digest over an empty
    set of paths is the hash of an empty byte string.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            h.update(Path(p).read_bytes())
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            pass
    return h.hexdigest()


def _git_commit() -> str:
    """Return the current git HEAD SHA or ``"no-git"`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "no-git"


def _default_serialiser(obj: Any) -> Any:
    """JSON default serialiser that handles common non-serialisable types."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):        # numpy array / pandas Series
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    Lightweight experiment tracker that stores all records in a single JSON
    file and provides DataFrame-based querying and comparison.

    Parameters
    ----------
    storage_path:
        Path to the JSON file used for persistence.  The file (and its
        parent directories) are created automatically on the first call to
        :meth:`save`.

    Examples
    --------
    >>> tracker = ExperimentTracker()
    >>> exp_id = tracker.start_experiment(config={"lr": 0.01}, strategy_names=["MomentumStrategy"])
    >>> tracker.log_result(exp_id, {"sharpe": 1.42, "cagr": 0.18})
    >>> tracker.finish_experiment(exp_id)
    >>> tracker.save()
    >>> df = tracker.list_experiments()
    """

    def __init__(self, storage_path: str = "data/experiments.json") -> None:
        self._storage_path: str = storage_path
        # dict[experiment_id -> experiment_record]
        self._experiments: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def start_experiment(
        self,
        config: dict[str, Any],
        strategy_names: list[str],
        data_files: list[str] | None = None,
    ) -> str:
        """
        Initialise a new experiment and return its unique experiment ID.

        Parameters
        ----------
        config:
            Full configuration dictionary for the experiment.
        strategy_names:
            List of strategy names used in this experiment.
        data_files:
            Optional list of data-file paths whose contents will be hashed
            to produce ``dataset_version``.  Pass ``None`` to skip.

        Returns
        -------
        str
            The UUID4 experiment ID.
        """
        experiment_id = str(uuid.uuid4())
        record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp": _utc_now(),
            "config_hash": _hash_dict(config),
            "dataset_version": _hash_files(data_files or []),
            "git_commit": _git_commit(),
            "strategy_set": list(strategy_names),
            "parameters": config,
            "results": {},
            "status": "running",
            "start_time": _utc_now(),
            "end_time": None,
            "duration_seconds": None,
        }
        self._experiments[experiment_id] = record
        return experiment_id

    def log_result(self, experiment_id: str, results: dict[str, Any]) -> None:
        """
        Merge *results* into the experiment record identified by
        *experiment_id*.

        Parameters
        ----------
        experiment_id:
            ID returned by :meth:`start_experiment`.
        results:
            Dictionary of metric names to values.  Multiple calls are
            merged (later keys overwrite earlier ones).

        Raises
        ------
        KeyError
            If *experiment_id* is not found.
        """
        record = self._get_record(experiment_id)
        record["results"].update(results)

    def finish_experiment(self, experiment_id: str) -> None:
        """
        Mark the experiment as finished and record its wall-clock duration.

        Parameters
        ----------
        experiment_id:
            ID returned by :meth:`start_experiment`.

        Raises
        ------
        KeyError
            If *experiment_id* is not found.
        """
        record = self._get_record(experiment_id)
        end_time_str = _utc_now()
        record["end_time"] = end_time_str
        record["status"] = "finished"

        try:
            start_dt = datetime.fromisoformat(record["start_time"])
            end_dt = datetime.fromisoformat(end_time_str)
            record["duration_seconds"] = (end_dt - start_dt).total_seconds()
        except (KeyError, ValueError):
            record["duration_seconds"] = None

    def fail_experiment(self, experiment_id: str, reason: str = "") -> None:
        """
        Mark the experiment as failed.

        Parameters
        ----------
        experiment_id:
            ID returned by :meth:`start_experiment`.
        reason:
            Optional human-readable failure reason stored in the record.
        """
        record = self._get_record(experiment_id)
        record["status"] = "failed"
        record["failure_reason"] = reason
        record["end_time"] = _utc_now()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """
        Return the full record for *experiment_id*.

        Returns
        -------
        dict
            A copy of the experiment record dict.

        Raises
        ------
        KeyError
            If *experiment_id* is not found.
        """
        return dict(self._get_record(experiment_id))

    def list_experiments(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of all experiments.

        The returned DataFrame contains one row per experiment.  Nested
        dicts (``parameters``, ``results``) are represented as JSON strings
        so that the frame is always flat.

        Returns
        -------
        pd.DataFrame
            Columns: experiment_id, timestamp, status, git_commit,
            config_hash, dataset_version, strategy_set, duration_seconds,
            parameters_json, results_json.
        """
        if not self._experiments:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for exp in self._experiments.values():
            row: dict[str, Any] = {
                "experiment_id": exp.get("experiment_id"),
                "timestamp": exp.get("timestamp"),
                "status": exp.get("status"),
                "git_commit": exp.get("git_commit"),
                "config_hash": exp.get("config_hash"),
                "dataset_version": exp.get("dataset_version"),
                "strategy_set": ", ".join(exp.get("strategy_set", [])),
                "duration_seconds": exp.get("duration_seconds"),
                "parameters_json": json.dumps(
                    exp.get("parameters", {}), default=_default_serialiser
                ),
                "results_json": json.dumps(
                    exp.get("results", {}), default=_default_serialiser
                ),
            }
            # Flatten top-level scalar results directly onto the row for
            # convenience (e.g. "sharpe", "cagr").
            for k, v in exp.get("results", {}).items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[f"result_{k}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        return df

    def compare_experiments(self, exp_ids: list[str]) -> pd.DataFrame:
        """
        Build a side-by-side comparison DataFrame for the given experiment IDs.

        Parameters
        ----------
        exp_ids:
            List of experiment IDs to compare.

        Returns
        -------
        pd.DataFrame
            Index = field name; one column per experiment ID.  Both the
            ``parameters`` and ``results`` sub-dicts are expanded into
            individual rows.

        Raises
        ------
        KeyError
            If any *experiment_id* is not found.
        """
        rows: dict[str, dict[str, Any]] = {}

        for eid in exp_ids:
            record = self._get_record(eid)
            flat: dict[str, Any] = {}

            # Top-level scalar fields
            for key in (
                "timestamp",
                "config_hash",
                "dataset_version",
                "git_commit",
                "status",
                "duration_seconds",
            ):
                flat[key] = record.get(key)

            # strategy_set as string
            flat["strategy_set"] = ", ".join(record.get("strategy_set", []))

            # Expand parameters
            for k, v in record.get("parameters", {}).items():
                flat[f"param.{k}"] = (
                    json.dumps(v, default=_default_serialiser)
                    if isinstance(v, (dict, list))
                    else v
                )

            # Expand results
            for k, v in record.get("results", {}).items():
                flat[f"result.{k}"] = v

            rows[eid] = flat

        # Build DataFrame: index = field name, columns = experiment IDs
        all_keys: list[str] = []
        seen: set[str] = set()
        for d in rows.values():
            for k in d:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        data = {eid: [rows[eid].get(k) for k in all_keys] for eid in exp_ids}
        return pd.DataFrame(data, index=all_keys)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """
        Persist all experiments to a JSON file.

        Parameters
        ----------
        path:
            Override the storage path set at construction time.  If
            ``None``, the constructor path is used.
        """
        target = Path(path or self._storage_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": "1.0",
            "saved_at": _utc_now(),
            "experiments": list(self._experiments.values()),
        }
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_default_serialiser)

    def load(self, path: str | None = None) -> None:
        """
        Load experiments from a JSON file, merging with any in-memory state.

        Existing in-memory records with the same ``experiment_id`` are
        **overwritten** by the file contents.

        Parameters
        ----------
        path:
            Override the storage path set at construction time.  If
            ``None``, the constructor path is used.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be parsed as valid JSON.
        """
        target = Path(path or self._storage_path)
        if not target.exists():
            raise FileNotFoundError(f"Experiment store not found: {target}")

        with target.open("r", encoding="utf-8") as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in experiment store: {exc}") from exc

        experiments = payload if isinstance(payload, list) else payload.get("experiments", [])
        for record in experiments:
            eid = record.get("experiment_id")
            if eid:
                self._experiments[eid] = record

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Auto-save on context-manager exit; does not suppress exceptions."""
        self.save()
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_record(self, experiment_id: str) -> dict[str, Any]:
        """Return the mutable record dict, raising KeyError if absent."""
        try:
            return self._experiments[experiment_id]
        except KeyError:
            raise KeyError(
                f"Experiment '{experiment_id}' not found. "
                f"Known IDs: {list(self._experiments.keys())}"
            ) from None

    def __len__(self) -> int:
        return len(self._experiments)

    def __repr__(self) -> str:
        return (
            f"ExperimentTracker("
            f"experiments={len(self._experiments)}, "
            f"storage_path='{self._storage_path}')"
        )
