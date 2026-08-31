"""Run provenance — git state, library versions, hardware — embedded in every artifact."""

from __future__ import annotations

import contextlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

_TRACKED_PACKAGES = (
    "torch",
    "numpy",
    "scipy",
    "scikit-learn",
    "ripser",
    "persim",
    "hydra-core",
    "omegaconf",
    "pandas",
    "matplotlib",
)


def _pkg_versions(names: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        with contextlib.suppress(PackageNotFoundError):
            out[name] = version(name)
    return out


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass
class EnvInfo:
    python_version: str
    platform: str
    git_commit: str | None
    git_dirty: bool | None
    torch_version: str | None
    cuda_available: bool
    mps_available: bool
    device_name: str | None
    packages: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_env_info() -> EnvInfo:
    torch_version: str | None = None
    cuda = mps = False
    device_name: str | None = None
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()
        if cuda:
            device_name = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch always present in practice
        pass

    status = _git("status", "--porcelain")
    return EnvInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=(bool(status) if status is not None else None),
        torch_version=torch_version,
        cuda_available=cuda,
        mps_available=mps,
        device_name=device_name,
        packages=_pkg_versions(),
    )
