"""Sparse weighted least squares for simulated trajectory-fitting examples."""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def rk4_step(rate, state, coefficients, dt):
    """Advance a batch of states by one classical Runge–Kutta step."""
    k1 = rate(state, coefficients)
    k2 = rate(state + dt / 2 * k1, coefficients)
    k3 = rate(state + dt / 2 * k2, coefficients)
    k4 = rate(state + dt * k3, coefficients)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def local_quadratic(readings, samples, target, dt):
    """Return the value and time derivative at a target sample."""
    offsets = (np.asarray(samples) - target) * dt
    design = np.polynomial.polynomial.polyvander(offsets, 2)
    return np.einsum("...tm,kt->...mk", readings, np.linalg.pinv(design)[:2])


def initialize_observed(readings, dt):
    """Initialize observed values and derivatives within each supplied window."""
    length = readings.shape[1]
    width = min(9, length)
    estimates = []
    for index in range(length):
        start = np.clip(index - width // 2, 0, length - width)
        samples = np.arange(start, start + width)
        estimates.append(local_quadratic(readings[:, samples], samples, index, dt))
    return np.stack(estimates, axis=1)


class TrajectoryFit:
    """Fit latent paths, optionally with coefficients shared across all paths.

    Measurements observe the leading state coordinates. The vectorized
    transition must preserve complex perturbations for complex-step derivatives.
    """

    def __init__(self, readings, step, measurement_scale, defect_scale, n_coefficients):
        self.readings = readings
        self.step = step
        self.measurement_scale = np.broadcast_to(measurement_scale, (readings.shape[-1],))
        self.defect_scale = np.asarray(defect_scale)
        self.n_coefficients = n_coefficients
        self.count, self.length, self.sensors = readings.shape
        self.dimension = len(self.defect_scale)
        shape = (self.count, self.length, self.dimension)
        self.n_states = np.prod(shape)
        state_ids = np.arange(self.n_states).reshape(shape)
        observation_ids = state_ids[..., :self.sensors].ravel()
        self.n_observations = observation_ids.size
        defect_rows = self.n_observations + np.arange(
            self.count * (self.length - 1) * self.dimension
        ).reshape(-1, self.dimension, 1)
        previous_ids = state_ids[:, :-1].reshape(-1, 1, self.dimension)
        next_ids = state_ids[:, 1:].reshape(-1, self.dimension, 1)
        columns = [np.broadcast_to(previous_ids, (*next_ids.shape[:2], self.dimension)), next_ids]
        if self.n_coefficients:
            coefficient_ids = self.n_states + np.arange(self.n_coefficients)
            columns.append(np.broadcast_to(coefficient_ids, (*next_ids.shape[:2], self.n_coefficients)))
        defect_columns = np.concatenate(columns, axis=-1)
        self.rows = np.r_[np.arange(self.n_observations), np.broadcast_to(defect_rows, defect_columns.shape).ravel()]
        self.columns = np.r_[observation_ids, defect_columns.ravel()]

    def residuals(self, states, coefficients):
        observation = (states[..., :self.sensors] - self.readings) / self.measurement_scale
        defect = (states[:, 1:] - self.step(states[:, :-1], coefficients)) / self.defect_scale
        return observation, defect

    def costs(self, states, coefficients):
        observation, defect = self.residuals(states, coefficients)
        return (observation ** 2).sum(axis=(1, 2)) + (defect ** 2).sum(axis=(1, 2))

    def linearize(self, states, coefficients):
        observation, defect = self.residuals(states, coefficients)
        previous = states[:, :-1].reshape(-1, self.dimension)
        delta = 1e-25j
        state_derivatives = [
            -self.step(previous + delta * basis, coefficients).imag / delta.imag / self.defect_scale
            for basis in np.eye(self.dimension)
        ]
        parts = [np.stack(state_derivatives, axis=-1),
                 np.broadcast_to(1 / self.defect_scale[:, None], (*previous.shape, 1))]
        if self.n_coefficients:
            coefficient_derivatives = [
                -self.step(previous, coefficients + delta * basis).imag / delta.imag / self.defect_scale
                for basis in np.eye(self.n_coefficients)
            ]
            parts.append(np.stack(coefficient_derivatives, axis=-1))
        data = np.r_[np.tile(1 / self.measurement_scale, self.count * self.length),
                     np.concatenate(parts, axis=-1).ravel()]
        residual = np.r_[observation.ravel(), defect.ravel()]
        jacobian = coo_matrix(
            (data, (self.rows, self.columns)),
            shape=(residual.size, self.n_states + self.n_coefficients),
        ).tocsc()
        return residual, jacobian

    def fit(self, initial_states, coefficients, max_updates=50, tolerance=1e-6):
        """Run damped Gauss–Newton with backtracking and scaled gradient stopping."""
        states = np.array(initial_states, dtype=float, copy=True)
        coefficients = np.array(coefficients, dtype=float, copy=True)
        costs = self.costs(states, coefficients)
        trace = [costs.mean() / self.length]
        for update in range(max_updates + 1):
            residual, jacobian = self.linearize(states, coefficients)
            normal = jacobian.T @ jacobian
            gradient = jacobian.T @ residual
            diagonal = np.maximum(normal.diagonal(), 1e-12)
            scaled = np.abs(gradient) / np.sqrt(diagonal)
            if self.n_coefficients:
                converged = np.array([scaled.max() < tolerance])
            else:
                converged = scaled.reshape(self.count, -1).max(axis=1) < tolerance
            if converged.all() or update == max_updates:
                break

            direction = spsolve(normal + diags(1e-8 * diagonal), -gradient)
            state_direction = direction[:self.n_states].reshape(states.shape)
            if self.n_coefficients:
                coefficient_direction = direction[self.n_states:]
                for exponent in range(24):
                    alpha = 0.5 ** exponent
                    next_states = states + alpha * state_direction
                    next_coefficients = coefficients + alpha * coefficient_direction
                    next_costs = self.costs(next_states, next_coefficients)
                    if next_costs.sum() < costs.sum():
                        states, coefficients, costs = next_states, next_coefficients, next_costs
                        break
                else:
                    break
            else:
                pending = ~converged
                for exponent in range(24):
                    next_states = states + 0.5 ** exponent * state_direction
                    next_costs = self.costs(next_states, coefficients)
                    accept = pending & (next_costs < costs)
                    states[accept], costs[accept] = next_states[accept], next_costs[accept]
                    pending[accept] = False
                    if not pending.any():
                        break
                if np.array_equal(pending, ~converged):
                    break
            trace.append(costs.mean() / self.length)

        return states, coefficients, {
            "cost_per_sample": np.array(trace),
            "max_scaled_gradient": scaled.max(),
            "converged_fraction": converged.mean(),
        }


def prefix_windows(readings, origins, width):
    """Extract windows ending at the requested sample indices, inclusive."""
    indices = np.asarray(origins)[:, None] + np.arange(1 - width, 1)
    if indices.min() < 0 or indices.max() >= readings.shape[1]:
        raise ValueError("Every origin must have a complete observation window.")
    return readings[:, indices]


def estimate_endpoints(windows, coefficients, step, measurement_scale, defect_scale, initialize):
    """Fit independent windows with a fixed transition and return their last states."""
    flat = windows.reshape(-1, *windows.shape[-2:])
    problem = TrajectoryFit(flat, step, measurement_scale, defect_scale, n_coefficients=0)
    states, _, diagnostic = problem.fit(initialize(flat, coefficients), coefficients)
    return states[:, -1].reshape(*windows.shape[:-2], problem.dimension), diagnostic


def rollout(states, coefficients, step, horizon):
    """Return the next ``horizon`` states without assimilating new measurements."""
    forecasts = []
    for _ in range(horizon):
        states = step(states, coefficients)
        forecasts.append(states)
    return np.stack(forecasts, axis=-2)


def rmse(prediction, target, axis=None):
    return np.sqrt(np.mean((prediction - target) ** 2, axis=axis))
