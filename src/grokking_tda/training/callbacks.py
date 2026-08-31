"""Callbacks — the engine's extension points, so the loop stays small."""

from __future__ import annotations

import time
from collections.abc import Iterable

from grokking_tda.utils.logging import get_logger

logger = get_logger(__name__)


class Callback:
    """Override the hooks you need; the engine calls all three."""

    def on_train_start(self, trainer) -> None:
        ...

    def on_step_end(self, trainer) -> None:
        ...

    def on_train_end(self, trainer) -> None:
        ...


class MetricLogger(Callback):
    def __init__(self, every: int) -> None:
        self.every = max(1, every)

    def on_train_start(self, trainer) -> None:
        trainer.record_metrics()  # step-0 baseline

    def on_step_end(self, trainer) -> None:
        if trainer.step % self.every == 0 or trainer.step == trainer.total_steps:
            trainer.record_metrics()


class SnapshotSaver(Callback):
    def __init__(self, steps: Iterable[int]) -> None:
        self.steps = set(steps)

    def on_train_start(self, trainer) -> None:
        if 0 in self.steps:
            trainer.save_snapshot()

    def on_step_end(self, trainer) -> None:
        if trainer.step in self.steps:
            trainer.save_snapshot()


class ConsoleProgress(Callback):
    def __init__(self, every: int = 1000) -> None:
        self.every = max(1, every)
        self._t0 = 0.0

    def on_train_start(self, trainer) -> None:
        self._t0 = time.perf_counter()

    def on_step_end(self, trainer) -> None:
        if trainer.step % self.every == 0 or trainer.step == trainer.total_steps:
            m = trainer.last_metrics
            if not m:
                return
            elapsed = max(time.perf_counter() - self._t0, 1e-9)
            rate = trainer.step / elapsed
            eta = (trainer.total_steps - trainer.step) / rate if rate > 0 else float("nan")
            logger.info(
                "step %6d/%d | train_acc %.3f test_acc %.3f | "
                "train_loss %.4f test_loss %.4f | %.0f it/s eta %.0fs",
                trainer.step,
                trainer.total_steps,
                m.get("train_acc", float("nan")),
                m.get("test_acc", float("nan")),
                m.get("train_loss", float("nan")),
                m.get("test_loss", float("nan")),
                rate,
                eta,
            )
