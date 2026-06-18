"""The training engine and its pluggable parts (callbacks, losses, optimizers)."""

from grokking_tda.training.callbacks import (
    Callback,
    ConsoleProgress,
    MetricLogger,
    SnapshotSaver,
)
from grokking_tda.training.engine import Trainer, default_callbacks
from grokking_tda.training.losses import build_loss
from grokking_tda.training.optimizers import OrthoGrad, build_optimizer
from grokking_tda.training.schedules import snapshot_steps

__all__ = [
    "Callback",
    "ConsoleProgress",
    "MetricLogger",
    "OrthoGrad",
    "SnapshotSaver",
    "Trainer",
    "build_loss",
    "build_optimizer",
    "default_callbacks",
    "snapshot_steps",
]

