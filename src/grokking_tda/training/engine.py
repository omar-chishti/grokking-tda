"""The step-centric training engine: take an optimizer step, then let callbacks observe."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
from torch import nn

from grokking_tda.artifacts.writer import ArtifactWriter
from grokking_tda.config.schema import ExperimentCfg
from grokking_tda.data.modular import ModularArithmeticData
from grokking_tda.training.callbacks import (
    Callback,
    ConsoleProgress,
    MetricLogger,
    SnapshotSaver,
)
from grokking_tda.training.losses import build_loss
from grokking_tda.training.optimizers import build_optimizer
from grokking_tda.training.schedules import snapshot_steps
from grokking_tda.training.trajectory import TrajectoryRecorder
from grokking_tda.utils.precision import tensor_norm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data: ModularArithmeticData,
        cfg: ExperimentCfg,
        device: torch.device,
        writer: ArtifactWriter,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.model = model.to(device)
        self.data = data.to(device)
        self.total_steps = cfg.train.steps
        self.loss_fn = build_loss(cfg.train.loss, cfg.train.loss_dtype)
        self.optimizer = build_optimizer(self.model.parameters(), cfg.train.optimizer)
        self.callbacks = list(callbacks) if callbacks is not None else default_callbacks(cfg)
        self.step = 0
        self.last_metrics: dict[str, float] = {}
        self._batch_gen = torch.Generator().manual_seed(cfg.seed + 1)

    def fit(self) -> Trainer:
        self.writer.append_event(
            {"event": "train_start", "total_steps": self.total_steps, "device": str(self.device)}
        )
        started = time.perf_counter()
        self._emit("on_train_start")
        while self.step < self.total_steps:
            for inputs, targets in self.data.iter_batches(
                self.cfg.train.batch_size, generator=self._batch_gen
            ):
                if self.step >= self.total_steps:
                    break
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(self.model(inputs), targets)
                loss.backward()
                self.optimizer.step()
                self.step += 1
                self._emit("on_step_end")
        self._emit("on_train_end")
        elapsed = time.perf_counter() - started
        self.writer.append_event(
            {
                "event": "train_end",
                "status": "completed",
                "steps": self.step,
                "wall_seconds": round(elapsed, 3),
                "steps_per_second": round(self.step / elapsed, 1) if elapsed > 0 else None,
                "final_metrics": self.last_metrics,
            }
        )
        return self

    def _emit(self, hook: str) -> None:
        for callback in self.callbacks:
            getattr(callback, hook)(self)

    @torch.no_grad()
    def _evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        self.model.eval()
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets).item()
        acc = (logits.argmax(dim=-1) == targets).float().mean().item()
        return loss, acc

    @torch.no_grad()
    def _weight_norm(self) -> float:
        return tensor_norm(self.model.parameters())

    def record_metrics(self) -> None:
        train_loss, train_acc = self._evaluate(self.data.train_inputs, self.data.train_targets)
        test_loss, test_acc = self._evaluate(self.data.test_inputs, self.data.test_targets)
        self.last_metrics = {
            "step": self.step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "weight_norm": self._weight_norm(),
        }
        self.writer.append_metric(self.last_metrics)

    @torch.no_grad()
    def save_snapshot(self) -> None:
        self.model.eval()
        reps = {"embedding": self.model.embedding_matrix().cpu().numpy()}
        if self.cfg.train.capture_representations:
            logits, cache = self.model.run_with_cache(
                self.data.inputs, names=[self.model.hidden_hook]
            )
            reps["logits"] = logits.detach().cpu().numpy()
            reps["hidden"] = cache[self.model.hidden_hook].detach().cpu().numpy()
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        self.writer.write_snapshot(self.step, state, reps)


def default_callbacks(cfg: ExperimentCfg) -> list[Callback]:
    steps = snapshot_steps(
        cfg.train.steps,
        cfg.train.n_snapshots,
        cfg.train.snapshot_schedule,
        cfg.train.dense_from,
        cfg.train.dense_to,
    )
    callbacks: list[Callback] = [
        MetricLogger(cfg.train.metric_every),
        SnapshotSaver(steps),
        ConsoleProgress(every=max(cfg.train.metric_every, cfg.train.steps // 20 or 1)),
    ]
    if cfg.train.trajectory_dim:
        callbacks.append(
            TrajectoryRecorder(cfg.train.trajectory_dim, cfg.train.trajectory_every, cfg.seed)
        )
    return callbacks
