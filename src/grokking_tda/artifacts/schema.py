"""Serialisable description of a run, embedded as ``manifest.json``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Bumped when a field changes meaning rather than when one is added; `reader.Run` refuses a
# manifest it does not carry, so a store written by a later version fails loudly.
SCHEMA_VERSION = 1


@dataclass
class Manifest:
    run_name: str
    config: dict[str, Any]  # fully-resolved ExperimentCfg (OmegaConf container)
    env: dict[str, Any]  # provenance: git, versions, hardware
    task_meta: dict[str, Any]  # TaskMeta (p, vocab_size, num_classes, seq_len, ...)
    created_at: str
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
