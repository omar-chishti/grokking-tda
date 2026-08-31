"""grokking_tda — grokking through topological data analysis. See docs/ARCHITECTURE.md."""

from __future__ import annotations

import os

# torch and scikit-learn load two OpenMP runtimes in one process, which deadlocks on Linux
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"

