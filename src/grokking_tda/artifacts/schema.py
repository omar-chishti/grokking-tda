"""Serialisable description of a run, embedded as ``manifest.json``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Manifest:
    run_name: str
    config: dict[str, Any]  # fully-resolved ExperimentCfg (OmegaConf container)
    env: dict[str, Any]  # provenance: git, versions, hardware
    task_meta: dict[str, Any]  # TaskMeta (p, vocab_size, num_classes, seq_len, ...)
    created_at: str
    schema_version: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
