"""Measurement correction, causal history processing and open-loop forecasts."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import torch
from torch import Tensor, nn


class JointEstimator(nn.Module):
    """Combine arbitrary differentiable dynamics with a fixed linear observation.

    H has shape [sensors, state components]. The corrector is two linear layers
    with Tanh and a zero-initialized output layer. Inputs to filter/loss have
    shape [independent sequences, measurement times, sensors].

    Optional fixed correction scales condition the neural network without
    normalizing the physical loss. An optional differentiable state_projection
    applies physical constraints after each measurement correction.
    """
    def __init__(self, dynamics: nn.Module, observation: Tensor, *, hidden_width: int = 64,
                 correction_input_scale: float = 1.0, correction_output_scale: float = 1.0,
                 state_projection: Callable[[Tensor], Tensor] | None = None):
        super().__init__()
        observation = torch.as_tensor(observation)
        if observation.ndim != 2 or min(observation.shape) < 1 or observation.is_complex():
            raise ValueError("H must be a nonempty real matrix [sensors, state components].")
        if not observation.is_floating_point():
            observation = observation.to(torch.get_default_dtype())
        if not torch.isfinite(observation).all():
            raise ValueError("H must contain finite values.")
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive.")
        for scale in (correction_input_scale, correction_output_scale):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Correction scales must be positive and finite.")
        observation = observation.detach().clone()
        self.state_dim = observation.shape[1]
        self.sensor_dim = observation.shape[0]
        self.dynamics = dynamics.to(device=observation.device, dtype=observation.dtype)
        self.register_buffer("H", observation)
        self.register_buffer("A", torch.linalg.pinv(observation))
        self.corrector = nn.Sequential(
            nn.Linear(2 * self.sensor_dim, hidden_width), nn.Tanh(),
            nn.Linear(hidden_width, self.state_dim),
        ).to(device=observation.device, dtype=observation.dtype)
        nn.init.zeros_(self.corrector[-1].weight)
        nn.init.zeros_(self.corrector[-1].bias)
        self.correction_input_scale = correction_input_scale
        self.correction_output_scale = correction_output_scale
        self.state_projection = state_projection

    def observe(self, state: Tensor) -> Tensor:
        return state @ self.H.T

    def correct(self, prior: Tensor, measurement: Tensor, learned: bool = True) -> Tensor:
        innovation = measurement - self.observe(prior)
        correction = innovation @ self.A.T
        if learned:
            correction = correction + self.correction_output_scale * self.corrector(
                torch.cat((innovation, measurement), -1) / self.correction_input_scale)
        posterior = prior + correction
        return posterior if self.state_projection is None else self.state_projection(posterior)

    def assimilate(self, measurement: Tensor, previous_posterior: Tensor | None = None,
                   learned: bool = True) -> Tensor:
        """Advance and correct once; None starts an independent sequence at zero."""
        if measurement.shape[-1] != self.sensor_dim:
            raise ValueError(f"Expected {self.sensor_dim} sensors.")
        prior = (measurement.new_zeros((*measurement.shape[:-1], self.state_dim))
                 if previous_posterior is None else self.dynamics(previous_posterior))
        return self.correct(prior, measurement, learned)

    def filter(self, measurements: Tensor, learned: bool = True) -> Tensor:
        """Process the entire supplied history with no state detachment."""
        if (measurements.ndim != 3 or min(measurements.shape[:2]) < 1
                or measurements.shape[2] != self.sensor_dim):
            raise ValueError(f"Supply a nonempty [B,T,{self.sensor_dim}] measurement tensor.")
        state = None
        corrected = []
        for t in range(measurements.shape[1]):
            state = self.assimilate(measurements[:, t], state, learned)
            corrected.append(state)
        return torch.stack(corrected, 1)

    def forecast(self, origin_states: Tensor, horizon: int) -> Tensor:
        """Predict with dynamics alone, accepting arbitrary leading batch dimensions."""
        if horizon < 1:
            raise ValueError("Forecast horizon must be at least one.")
        state = origin_states
        future = []
        for _ in range(horizon):
            state = self.dynamics(state)
            future.append(state)
        return torch.stack(future, -2)

    def loss(self, measurements: Tensor, origins: Sequence[int],
             horizon: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return objective, future MSE and diagnostic current MSE.

        The objective is future MSE alone. Current disagreement with already
        assimilated noisy measurements is recorded once per history time and
        does not contribute to training or checkpoint selection.
        """
        if measurements.ndim != 3:
            raise ValueError("Supply a [B,T,M] measurement tensor.")
        origins = tuple(origins)
        if (not origins or min(origins) < 0
                or max(origins) + horizon >= measurements.shape[1]):
            raise ValueError("Each origin must have a history and a complete future target.")
        if len(set(origins)) != len(origins):
            raise ValueError("Forecast origins must be distinct.")
        history = measurements[:, :max(origins) + 1]
        corrected = self.filter(history)
        current_loss = (self.observe(corrected) - history).square().mean()
        future_states = self.forecast(corrected[:, list(origins)], horizon)
        targets = torch.stack([measurements[:, o + 1:o + 1 + horizon] for o in origins], 1)
        future_loss = (self.observe(future_states) - targets).square().mean()
        return future_loss, future_loss, current_loss
