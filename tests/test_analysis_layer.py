"""The reduction layer is verified by its committed output rather than by unit tests, which
leaves one gap a committed CSV cannot cover: a module that no longer imports. `mypy` is scoped
to the installed package and `ruff` parses rather than compiles, so a driver can be broken for a
whole session before the next run of it says so.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _modules(package: str) -> list[str]:
    path = ROOT / package.replace(".", "/")
    found = [f"{package}.{m.name}" for m in pkgutil.iter_modules([str(path)]) if not m.ispkg]
    return sorted(found)


@pytest.mark.parametrize(
    "name", _modules("analysis") + _modules("analysis.figures") + _modules("sidequest")
)
def test_every_driver_imports(name: str) -> None:
    importlib.import_module(name)
