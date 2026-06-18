"""The artifact store: the on-disk contract between training and analysis.

A run directory is self-describing::

    <output_root>/<run_name>/
        manifest.json          # resolved config + env provenance + task meta
        metrics.jsonl          # one JSON object per recorded step (scalars)
        events.jsonl           # lifecycle/timing/failure events (observability)
        snapshots/
            index.json         # list of fully-saved snapshot steps
            step_00000000/
                weights.pt          # authoritative model state
                representations.npz # cached point clouds (at least 'embedding')
                meta.json
            ...

``ArtifactWriter`` produces this; ``Run``/``Snapshot`` consume it. Analysis code
depends only on these classes, never on the training engine. JSON files are written
atomically and the snapshot index is written last, so a killed run never leaves a
half-written file that breaks a reader.
"""

from grokking_tda.artifacts.reader import Run, Snapshot
from grokking_tda.artifacts.schema import Manifest
from grokking_tda.artifacts.writer import ArtifactWriter, prepare_run_dir

__all__ = ["ArtifactWriter", "Manifest", "Run", "Snapshot", "prepare_run_dir"]
