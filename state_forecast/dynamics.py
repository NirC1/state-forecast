"""Differentiable integration of a user-supplied state-rate model."""
from __future__ import annotations

import math

from torch import Tensor, nn


class ODEDynamics(nn.Module):
    """Turn ``rate_model(x)`` into a one-sample state transition Phi.

    The supplied PyTorch module must return a rate with the same shape as x.
    Its parameters are learned by the joint optimizer. A discrete dynamics
    module can instead be passed directly to JointEstimator without this wrapper.
    """
    def __init__(self, rate_model: nn.Module, dt: float, integrator: str = "rk4"):
        super().__init__()
        if not math.isfinite(dt) or dt <= 0 or integrator not in {"rk4", "euler"}:
            raise ValueError("Use positive finite dt and integrator 'rk4' or 'euler'.")
        self.rate_model = rate_model
        self.dt = dt
        self.integrator = integrator

    def rate(self, state: Tensor) -> Tensor:
        return self.rate_model(state)

    def forward(self, state: Tensor) -> Tensor:
        h = self.dt
        k1 = self.rate(state)
        if self.integrator == "euler":
            return state + h * k1
        k2 = self.rate(state + h * k1 / 2)
        k3 = self.rate(state + h * k2 / 2)
        k4 = self.rate(state + h * k3)
        return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

