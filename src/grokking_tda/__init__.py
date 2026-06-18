"""grokking_tda — a research framework for studying grokking via topological data analysis.

The package is organised as three decoupled layers (see ``docs/ARCHITECTURE.md``):

1. **train**    (``training`` + ``models`` + ``data``) — a step-centric engine that
   trains a model and emits immutable, self-describing run artifacts.
2. **artifacts** (``artifacts``) — the on-disk contract: a manifest, step-indexed
   scalar metrics, and snapshots (weights + cached representations).
3. **analyse**  (``analysis`` + ``tda`` + ``baselines`` + ``evaluation`` + ``plotting``) —
   offline consumers that turn artifacts into observables, comparisons, and figures.

The layers communicate *only* through the artifact store, so expensive GPU training
runs once while cheap CPU analysis iterates many times.
"""

from __future__ import annotations

import os

# torch (Intel OpenMP) and scikit-learn/scipy (LLVM OpenMP) can load two OpenMP
# runtimes in one process; on Linux this can deadlock. Set defensively at import,
# before torch/sklearn are imported by submodules.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"

