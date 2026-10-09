"""Measurement-only joint Adam optimization; callers own data and artifacts."""
from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import dataclass
import math
import time

import torch
from torch import Tensor

from .estimator import JointEstimator


@dataclass(frozen=True)
class TrainingConfig:
    origins: tuple[int, ...]
    horizon: int
    updates: int = 1000
    batch_size: int = 32
    dynamics_lr: float = 0.001
    corrector_lr: float = 0.001
    current_weight: float = 0.1
    validate_every: int = 100
    seed: int = 7
    max_grad_norm: float | None = 20.0
    train_score_sequences: int = 64
    freeze_after_training: bool = True

    def __post_init__(self):
        if self.updates < 0 or min(self.horizon, self.batch_size, self.validate_every,
                                   self.train_score_sequences) < 1:
            raise ValueError("Use nonnegative updates and positive horizon, batch and validation sizes.")
        if not self.origins or min(self.origins) < 0 or len(set(self.origins)) != len(self.origins):
            raise ValueError("Supply distinct nonnegative forecast origins.")
        if any(not math.isfinite(lr) or lr <= 0 for lr in (self.dynamics_lr, self.corrector_lr)):
            raise ValueError("Learning rates must be positive and finite.")
        if not math.isfinite(self.current_weight) or self.current_weight < 0:
            raise ValueError("current_weight must be nonnegative and finite.")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive and finite, or None.")


@dataclass(frozen=True)
class ValidationProgress:
    """Validation record and current best snapshot, supplied to an optional callback."""
    record: dict
    best_step: int
    best_state: dict[str, Tensor]


@dataclass
class TrainingResult:
    history: list[dict]
    best_step: int
    best_validation_loss: float
    best_state: dict[str, Tensor]


def fit(model: JointEstimator, train_measurements: Tensor, validation_measurements: Tensor,
        config: TrainingConfig, *, on_validation: Callable[[ValidationProgress], None] | None = None
        ) -> TrainingResult:
    """Fit theta and phi jointly, restore the best validation model and optionally freeze.

    Only measurement tensors enter this API. Batch members are independent
    sequences, sampled with replacement by a local seeded generator. This
    function does not generate data, alter global seeds/thread counts or write
    files. Use on_validation to record system-specific parameters or checkpoints.
    Call model.requires_grad_(True) before continuing a previously frozen model.
    """
    for measurements in (train_measurements, validation_measurements):
        if (measurements.ndim != 3 or measurements.shape[0] < 1
                or measurements.shape[2] != model.sensor_dim
                or max(config.origins) + config.horizon >= measurements.shape[1]):
            raise ValueError("Measurement tensors must contain independent sequences and complete targets.")
        if not torch.isfinite(measurements).all():
            raise ValueError("Training and validation measurements must be finite.")
    groups = [
        {"params": [p for p in model.dynamics.parameters() if p.requires_grad], "lr": config.dynamics_lr},
        {"params": [p for p in model.corrector.parameters() if p.requires_grad], "lr": config.corrector_lr},
    ]
    if not any(group["params"] for group in groups):
        raise ValueError("The model has no trainable parameters; unfreeze it before calling fit.")
    optimizer = torch.optim.Adam(groups)
    generator = torch.Generator(device=train_measurements.device).manual_seed(config.seed)
    history, best_loss, best_step = [], float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.monotonic()
    gradient_norm = 0.0
    for step in range(config.updates + 1):
        if step:
            model.train()
            indices = torch.randint(len(train_measurements), (config.batch_size,),
                                    generator=generator, device=train_measurements.device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model.loss(train_measurements[indices], config.origins,
                                    config.horizon, config.current_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at update {step}.")
            loss.backward()
            limit = config.max_grad_norm if config.max_grad_norm is not None else float("inf")
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), limit,
                                                                 error_if_nonfinite=True))
            optimizer.step()
        if step % config.validate_every == 0 or step == config.updates:
            model.eval()
            with torch.no_grad():
                train_scores = model.loss(train_measurements[:config.train_score_sequences],
                                          config.origins, config.horizon, config.current_weight)
                val_scores = model.loss(validation_measurements, config.origins,
                                        config.horizon, config.current_weight)
            if not all(torch.isfinite(score) for score in (*train_scores, *val_scores)):
                raise RuntimeError(f"Non-finite validation or diagnostic loss at update {step}.")
            record = {"step": step, "elapsed_seconds": time.monotonic() - started,
                      "train_loss": float(train_scores[0]), "train_future": float(train_scores[1]),
                      "train_current": float(train_scores[2]), "val_loss": float(val_scores[0]),
                      "val_future": float(val_scores[1]), "val_current": float(val_scores[2]),
                      "gradient_norm_before_clip": gradient_norm}
            history.append(record)
            if record["val_loss"] < best_loss:
                best_loss, best_step = record["val_loss"], step
                best_state = copy.deepcopy(model.state_dict())
            if on_validation is not None:
                on_validation(ValidationProgress(record, best_step, best_state))
    model.load_state_dict(best_state)
    model.eval()
    if config.freeze_after_training:
        model.requires_grad_(False)
    return TrainingResult(history, best_step, best_loss, best_state)

