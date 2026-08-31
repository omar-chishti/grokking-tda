"""The artifact store: the on-disk contract between training and analysis.

    <output_root>/<run_name>/
        manifest.json          resolved config, environment provenance, task meta
        metrics.jsonl          one record per logged step
        events.jsonl           lifecycle, timing, failures
        snapshots/index.json   the steps that are fully written
        snapshots/step_*/      weights.pt, representations.npz, meta.json

JSON is written atomically and the index last, so a killed run leaves a reader nothing partial.
"""

from grokking_tda.artifacts.reader import Run, Snapshot
from grokking_tda.artifacts.schema import Manifest
from grokking_tda.artifacts.writer import ArtifactWriter, prepare_run_dir

__all__ = ["ArtifactWriter", "Manifest", "Run", "Snapshot", "prepare_run_dir"]
