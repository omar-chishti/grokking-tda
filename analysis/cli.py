"""The command line every driver in this directory shares: ``--root`` and ``--out``."""

from __future__ import annotations

import argparse
from pathlib import Path

RUN_ROOT = Path("results/raw")
OUT_ROOT = Path("results/processed/thesis")


def parser(doc: str | None, *, root: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(doc or "").splitlines()[0])
    if root:
        ap.add_argument("--root", type=Path, default=RUN_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    return ap
