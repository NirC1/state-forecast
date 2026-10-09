"""Lorenz example data, candidate terms, evaluation, and command-line runner.

The reusable filter, forecast, integration and training loop live in state_forecast.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from state_forecast import JointEstimator, ODEDynamics, TrainingConfig, ValidationProgress, fit

TRUE_COEFFICIENTS = np.array([-10.0, 10.0, 28.0, -1.0, -8.0 / 3.0])
COEFFICIENT_NAMES = ["u in du/dt", "v in du/dt", "u in dv/dt", "v in dv/dt", "w in dw/dt"]


@dataclass
class Config:
    seed: int = 7
    dt: float = 0.01
    noise_std: float = 0.3
    integrator: str = "rk4"
    train_sequences: int = 256
    val_sequences: int = 64
    test_sequences: int = 64
    test_seed: int = 101
    sensors: int = 2
    batch_size: int = 48
    updates: int = 1600
    origins: tuple[int, ...] = (60, 80, 100)
    horizon: int = 40
    current_weight: float = 0.1
    hidden_width: int = 64
    dynamics_lr: float = 0.04
    corrector_lr: float = 0.002
    validate_every: int = 100


class LorenzRate(nn.Module):
    """Example candidate family: five signed coefficients and fixed quadratic terms."""
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([-6.0, 6.0, 18.0, -0.5, -1.5]))

    def forward(self, state: Tensor) -> Tensor:
        u, v, w = state.unbind(-1)
        a, b, c, d, e = self.coefficients.unbind()
        return torch.stack((a * u + b * v, c * u + d * v - u * w, u * v + e * w), -1)


class LorenzDynamics(ODEDynamics):
    """Lorenz-specific rate model inside the library's general ODE integrator."""
    def __init__(self, dt: float = 0.01, integrator: str = "rk4"):
        super().__init__(LorenzRate(), dt, integrator)

    @property
    def coefficients(self):
        return self.rate_model.coefficients


def make_model(config: Config, observation: Tensor | None = None) -> JointEstimator:
    if observation is None:
        if config.sensors not in {1, 2, 3}:
            raise ValueError("Choose one (u), two (u,v), or three (u,v,w) sensors.")
        observation = torch.eye(3)[:config.sensors]
    return JointEstimator(LorenzDynamics(config.dt, config.integrator), observation,
                          hidden_width=config.hidden_width,
                          correction_input_scale=20.0, correction_output_scale=10.0)



def load_checkpoint(path: Path, *, freeze: bool = True):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = Config(**saved["config"])
    model = make_model(config)
    model.load_state_dict(saved["model"])
    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model, config, saved


def training_config(config: Config) -> TrainingConfig:
    return TrainingConfig(origins=tuple(config.origins), horizon=config.horizon,
                          updates=config.updates, batch_size=config.batch_size,
                          dynamics_lr=config.dynamics_lr, corrector_lr=config.corrector_lr,
                          current_weight=config.current_weight,
                          validate_every=config.validate_every, seed=config.seed + 20)


def simulate(count: int, steps: int, config: Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent Lorenz trajectories; fine-step RK4 differs from model solver."""
    rng = np.random.default_rng(seed)
    state = np.column_stack((rng.uniform(-15, 15, count), rng.uniform(-15, 15, count),
                             rng.uniform(10, 35, count)))
    def rate(x):
        u, v, w = x.T
        return np.stack((10 * (v - u), u * (28 - w) - v, u * v - (8 / 3) * w), -1)
    def advance(x):
        h = config.dt / 4
        for _ in range(4):
            k1 = rate(x)
            k2 = rate(x + h * k1 / 2)
            k3 = rate(x + h * k2 / 2)
            k4 = rate(x + h * k3)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x
    for _ in range(round(3.0 / config.dt)):
        state = advance(state)
    true_states = [state]
    for _ in range(steps - 1):
        state = advance(state)
        true_states.append(state)
    truth = np.stack(true_states, 1).astype(np.float32)
    measurements = truth[..., :config.sensors] + rng.normal(
        0, config.noise_std, (count, steps, config.sensors))
    return truth, measurements.astype(np.float32)


def train(config: Config, output: Path, warm_start: Path | None = None):
    """Prepare Lorenz measurements and delegate optimization to the library."""
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    length = max(config.origins) + config.horizon + 1
    _, train_y = simulate(config.train_sequences, length, config, config.seed + 1)
    _, val_y = simulate(config.val_sequences, length, config, config.seed + 2)
    model = make_model(config)
    initial = copy.deepcopy(model.state_dict())
    if warm_start is not None:
        saved = torch.load(warm_start, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        initial = saved["initial_model"]
    settings = training_config(config)
    history = []

    def record(progress: ValidationProgress):
        row = progress.record
        row["coefficients"] = model.dynamics.rate_model.coefficients.detach().tolist()
        history.append(row)
        print(f"update {row['step']:4d} | train {row['train_loss']:.4f} | "
              f"val {row['val_loss']:.4f} | coefficients "
              f"{np.round(row['coefficients'], 3)} | {row['elapsed_seconds']:.1f}s", flush=True)
        (output / "training.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": progress.best_state, "initial_model": initial,
                    "config": asdict(config), "best_step": progress.best_step,
                    "warm_start": str(warm_start) if warm_start else None,
                    "library": "state_forecast", "training_config": asdict(settings)},
                   output / "checkpoint.pt")

    result = fit(model, torch.from_numpy(train_y), torch.from_numpy(val_y), settings,
                 on_validation=record)
    return model, initial, result.history, result.best_step


@torch.no_grad()
def evaluate(model: JointEstimator, initial: dict, config: Config, output: Path):
    """Independent test trajectories, persistence, and a fixed-correction ablation."""
    true_states, measurements = simulate(config.test_sequences, 1001, config, config.test_seed)
    y = torch.from_numpy(measurements)
    filtered = model.filter(y)
    fixed_filtered = model.filter(y, learned=False)
    origin = 100  # Entire supplied prefix 0..100 corresponds to local -100..0.
    evaluation_origins = (80, 100, 120, 140)
    eval_horizon = 200
    forecast = model.forecast(filtered[:, list(evaluation_origins)], eval_horizon)
    fixed_forecast = model.forecast(fixed_filtered[:, list(evaluation_origins)], eval_horizon)
    truth = np.stack([true_states[:, o + 1:o + 1 + eval_horizon] for o in evaluation_origins], 1)
    noisy_targets = np.stack([measurements[:, o + 1:o + 1 + eval_horizon] for o in evaluation_origins], 1)
    persistence = np.stack([np.repeat(measurements[:, o:o + 1], eval_horizon, axis=1)
                            for o in evaluation_origins], 1)
    forecast = forecast.numpy()
    fixed_forecast = fixed_forecast.numpy()
    def rmse(x):
        return float(np.sqrt(np.mean(np.square(x))))
    sensors = config.sensors
    lead_rmse = np.sqrt(np.mean((forecast[..., :sensors] - truth[..., :sensors]) ** 2, axis=(0, 1, 3)))
    fixed_lead_rmse = np.sqrt(np.mean((fixed_forecast[..., :sensors] - truth[..., :sensors]) ** 2, axis=(0, 1, 3)))
    persistence_rmse = np.sqrt(np.mean((persistence - truth[..., :sensors]) ** 2, axis=(0, 1, 3)))
    start = 80
    coeff = model.dynamics.rate_model.coefficients.numpy().astype(float)
    metrics = {
        "test_sequences": config.test_sequences,
        "sensors": sensors,
        "test_forecast_origins_per_sequence": len(evaluation_origins),
        "noise_std": config.noise_std,
        "state_rmse_after_0.8_time_units": np.sqrt(np.mean(
            (filtered.numpy()[:, start:] - true_states[:, start:]) ** 2, axis=(0, 1))).tolist(),
        "fixed_corrector_state_rmse_after_0.8_time_units": np.sqrt(np.mean(
            (fixed_filtered.numpy()[:, start:] - true_states[:, start:]) ** 2, axis=(0, 1))).tolist(),
        "sensor_noise_rmse": rmse(measurements[:, start:] - true_states[:, start:, :sensors]),
        "forecast_sensor_rmse_first_0.4_time_units": rmse(forecast[:, :, :40, :sensors] - truth[:, :, :40, :sensors]),
        "forecast_noisy_target_rmse_first_0.4_time_units": rmse(forecast[:, :, :40, :sensors] - noisy_targets[:, :, :40]),
        "persistence_sensor_rmse_first_0.4_time_units": rmse(persistence[:, :, :40] - truth[:, :, :40, :sensors]),
        "fixed_corrector_sensor_rmse_first_0.4_time_units": rmse(fixed_forecast[:, :, :40, :sensors] - truth[:, :, :40, :sensors]),
        "learned_coefficients": coeff.tolist(),
        "true_coefficients": TRUE_COEFFICIENTS.tolist(),
        "coefficient_relative_error_percent": (100 * np.abs((coeff - TRUE_COEFFICIENTS) / TRUE_COEFFICIENTS)).tolist(),
        "forecast_leads": {f"{lead * config.dt:.2f}": float(lead_rmse[lead - 1])
                           for lead in [1, 10, 40, 100, 200]},
        "training_truth_used": False,
        "solver": config.integrator,
        "structural_prior": "Lorenz term support, with fixed -u*w and +u*v coefficients",
        "units": "Lorenz nondimensional state and time units",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    initial_model = make_model(config)
    initial_model.load_state_dict(initial)
    initial_filtered = initial_model.filter(y[:, :origin + 1]).numpy()
    np.savez_compressed(output / "evaluation.npz", truth=true_states, measurements=measurements,
                        filtered=filtered.numpy(), fixed_filtered=fixed_filtered.numpy(),
                        forecasts=forecast, forecast_truth=truth, origin=origin,
                        initial_filtered=initial_filtered, evaluation_origins=evaluation_origins,
                        lead_rmse=lead_rmse, fixed_lead_rmse=fixed_lead_rmse,
                        persistence_rmse=persistence_rmse)
    with (output / "forecast_errors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lead_time", "joint_sensor_rmse", "fixed_corrector_sensor_rmse", "persistence_sensor_rmse"])
        writer.writerows(zip(np.arange(1, eval_horizon + 1) * config.dt, lead_rmse,
                            fixed_lead_rmse, persistence_rmse))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--integrator", choices=["rk4", "euler"], default="rk4")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--dynamics-lr", type=float, default=0.04)
    parser.add_argument("--corrector-lr", type=float, default=0.002)
    parser.add_argument("--test-seed", type=int, default=101)
    parser.add_argument("--sensors", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--warm-start", type=Path, help="Continue learning from a saved best model, with a fresh Adam optimizer.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--evaluate-only", action="store_true", help="Rebuild metrics/figures from checkpoint.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        torch.set_num_threads(1)
        model, config, checkpoint = load_checkpoint(args.output / "checkpoint.pt")
        initial, best_step = checkpoint["initial_model"], checkpoint["best_step"]
        warm_start_provenance = checkpoint["warm_start"]
        log = json.loads((args.output / "training.json").read_text())
    else:
        config = Config(seed=args.seed, updates=args.updates, integrator=args.integrator,
                        horizon=args.horizon, dynamics_lr=args.dynamics_lr,
                        corrector_lr=args.corrector_lr, test_seed=args.test_seed,
                        sensors=args.sensors)
        (args.output / "config.json").write_text(json.dumps(asdict(config), indent=2))
        model, initial, log, best_step = train(config, args.output, args.warm_start)
        warm_start_provenance = str(args.warm_start) if args.warm_start else None
    metrics = evaluate(model, initial, config, args.output)
    metrics["best_validation_update"] = best_step
    metrics["runtime_versions"] = {"torch": torch.__version__, "numpy": np.__version__}
    metrics["test_simulation_seed"] = config.test_seed
    metrics["warm_start_checkpoint"] = warm_start_provenance
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    from examples.lorenz_plots import make_figures
    make_figures(args.output, config)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
