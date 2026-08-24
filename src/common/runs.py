"""
Run registry — what was trained, on what, and how well it did.

The previous setup answered none of those questions without unpickling a
joblib. Model artifacts and PNGs were committed to git as opaque binaries,
`metadata_v3.joblib` doubled as the registry, and the version tag was a hand-
typed "v1"/"v2"/"v3" that carried no information about what changed between
them. Reproducing a number from last week meant guessing.

Each run writes `runs/<timestamp>__<name>/run.json` with parameters, metrics,
the feature list, a content hash of the training data, and the git SHA. That
file is small, diffable, greppable, and committable; the model binaries it
points at are not committed.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

RUNS_DIR = Path(config.PROJECT_ROOT) / "runs"


def git_sha() -> str | None:
    """Current commit, or None outside a repo. Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def frame_hash(df, columns: list[str] | None = None) -> str:
    """
    Content hash of a training matrix.

    Hashes shape, column names, and the pandas row hashes, so two runs that
    claim the same data really saw the same data. Catches the failure mode
    where an ingestor backfilled rows between two runs and the metric
    difference gets attributed to a modelling change.
    """
    import pandas as pd

    cols = columns if columns is not None else list(df.columns)
    subset = df[[c for c in cols if c in df.columns]]
    h = hashlib.sha256()
    h.update(str(subset.shape).encode())
    h.update("|".join(map(str, subset.columns)).encode())
    h.update(pd.util.hash_pandas_object(subset, index=False).values.tobytes())
    return h.hexdigest()[:16]


@dataclass
class Run:
    """One training run. Use `start_run` rather than constructing directly."""

    name: str
    directory: Path
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str | None = None

    def log_params(self, **kwargs) -> None:
        self.params.update(kwargs)

    def log_metrics(self, prefix: str | None = None, **kwargs) -> None:
        for key, value in kwargs.items():
            full = f"{prefix}.{key}" if prefix else key
            # numpy scalars are not JSON-serializable; normalize on the way in
            # so a run cannot fail to save after the expensive part is done.
            self.metrics[full] = float(value) if hasattr(value, "item") else value

    def log_data(self, **kwargs) -> None:
        self.data.update(kwargs)

    def artifact_path(self, filename: str) -> Path:
        """Path inside the run directory; registers the artifact."""
        self.directory.mkdir(parents=True, exist_ok=True)
        if filename not in self.artifacts:
            self.artifacts.append(filename)
        return self.directory / filename

    def finish(self) -> Path:
        """Write run.json. Returns its path."""
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "run.json"
        payload = {
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "python": platform.python_version(),
            "params": self.params,
            "data": self.data,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path


def start_run(name: str, **params) -> Run:
    """Begin a run. Directory name sorts chronologically."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = RUNS_DIR / f"{stamp}__{name}"
    run = Run(
        name=name,
        directory=directory,
        params=dict(params),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    run.directory.mkdir(parents=True, exist_ok=True)
    return run


def list_runs(limit: int = 20) -> list[dict]:
    """Recent runs, newest first, as parsed run.json payloads."""
    if not RUNS_DIR.exists():
        return []
    out = []
    for directory in sorted(RUNS_DIR.iterdir(), reverse=True):
        manifest = directory / "run.json"
        if manifest.exists():
            try:
                out.append(json.loads(manifest.read_text()))
            except json.JSONDecodeError:
                continue
        if len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    for entry in list_runs():
        metrics = entry.get("metrics", {})
        headline = metrics.get("test.log_loss", metrics.get("log_loss", "—"))
        dirty = " (dirty)" if entry.get("git_dirty") else ""
        print(f"{entry['started_at']}  {entry['name']:<28} "
              f"log_loss={headline}  {str(entry.get('git_sha'))[:8]}{dirty}")
