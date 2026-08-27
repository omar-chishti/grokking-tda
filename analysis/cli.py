"""The command line every driver in this directory shares.

Each module answers one question about the run bank and writes the answer under one
output directory. Only the question differs between them, so only the question is
written per module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

RUN_ROOT = Path("results/raw")
OUT_ROOT = Path("results/processed/thesis")


def parser(doc: str | None, *, root: bool = True) -> argparse.ArgumentParser:
    """A parser carrying ``--root``/``--out``, described by the module's opening line."""
    ap = argparse.ArgumentParser(description=(doc or "").splitlines()[0])
    if root:
        ap.add_argument("--root", type=Path, default=RUN_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    return ap
