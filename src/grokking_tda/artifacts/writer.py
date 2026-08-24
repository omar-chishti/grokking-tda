"""Write run artifacts to disk in the schema documented in ``artifacts/__init__.py``."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grokking_tda.artifacts.schema import Manifest


def prepare_run_dir(run_dir: str | Path, overwrite: bool = False) -> Path:
    """Guard against silently reusing a run directory that already holds a run.

    The writer truncates the append-only logs on start, so rerunning into an existing
    directory without cleanup would pair fresh metrics with stale ``snapshots/step_*``
    directories — a mixed-run artifact. Refuse unless ``overwrite``; on overwrite,
    remove every prior artifact first.
    """
    run_dir = Path(run_dir)
    has_run = (run_dir / "manifest.json").exists() or (run_dir / "snapshots").exists()
    if not has_run:
        return run_dir
    if not overwrite:
        raise FileExistsError(
            f"{run_dir} already contains a run (manifest.json/snapshots). "
            "Pass overwrite=true to replace it, or change run_name."
        )
    for name in ("manifest.json", "metrics.jsonl", "events.jsonl"):
        (run_dir / name).unlink(missing_ok=True)
    for sub in ("snapshots", "analysis", "figures"):
        shutil.rmtree(run_dir / sub, ignore_errors=True)
    return run_dir


class ArtifactWriter:
    """Creates a run directory and appends metrics / snapshots as training proceeds."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.snapshots_dir = self.run_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.events_path = self.run_dir / "events.jsonl"
        self._snapshot_steps: list[int] = []
        # Truncate append-only logs on (re)start so a rerun never appends to stale data.
        self.metrics_path.write_text("")
        self.events_path.write_text("")

    def write_manifest(self, manifest: Manifest) -> None:
        self._dump(self.run_dir / "manifest.json", manifest.to_dict())

    def append_metric(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.metrics_path, record)

    def append_event(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        self._append_jsonl(self.events_path, record)

    def write_snapshot(
        self,
        step: int,
        state_dict: dict[str, torch.Tensor],
        representations: dict[str, np.ndarray],
    ) -> None:
        snap_dir = self.snapshots_dir / f"step_{step:08d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, snap_dir / "weights.pt")
        np.savez_compressed(snap_dir / "representations.npz", **representations)
        self._dump(snap_dir / "meta.json", {"step": step})
        self._snapshot_steps.append(step)
        # Written last and atomically: the index only ever lists fully-saved snapshots.
        self._dump(self.snapshots_dir / "index.json", {"steps": sorted(set(self._snapshot_steps))})

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _dump(path: Path, payload: dict[str, Any]) -> None:
        # Write to a temp file then rename: a reader never sees a partial JSON file.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
