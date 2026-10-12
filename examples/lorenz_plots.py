"""Reproducible scientific figures from saved evaluation arrays."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE, ORANGE, GRAY, OLIVE = "#2563eb", "#c87516", "#8b95a5", "#707b32"


def make_figures(output: Path, config):
    data = np.load(output / "evaluation.npz")
    log = json.loads((output / "training.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    sensors = getattr(config, "sensors", 2)
    labels = tuple(name if index < sensors else f"{name} (hidden)"
                   for index, name in enumerate(("u", "v", "w")))
    training_title = ("Refinement training objective" if metrics.get("warm_start_checkpoint")
                      else "Joint training objective")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#293343", "text.color": "#293343",
                         "axes.edgecolor": "#b3bac4", "grid.color": "#e4e7ec",
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    truth, measurements, filtered = data["truth"], data["measurements"], data["filtered"]
    sample = 0  # Predetermined first held-out sequence, never selected by performance.
    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    # Both panels use the same ranges and camera; start after the declared warm-up.
    fig = plt.figure(figsize=(12, 5.6), layout="constrained")
    for index, (states, color, title) in enumerate([
            (truth[sample, 80:], BLUE, "True Lorenz trajectory"),
            (filtered[sample, 80:], ORANGE, f"Estimated from {sensors} noisy sensors")], 1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.plot(*states.T, color=color, linewidth=0.9)
        ax.set(xlabel="u", ylabel="v", zlabel="w", xlim=(-22, 22),
               ylim=(-30, 30), zlim=(0, 52), title=title)
        ax.view_init(elev=24, azim=-63)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
    fig.suptitle("Reconstructing the Lorenz attractor", fontsize=17, fontweight="bold")
    save(fig, "attractor")

    t = np.arange(truth.shape[1]) * config.dt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for component, ax in enumerate(axes):
        if component < sensors:
            ax.scatter(t[::4], measurements[sample, ::4, component], s=8,
                       color=GRAY, alpha=0.45, label="Noisy measurements", rasterized=True)
        ax.plot(t, truth[sample, :, component], color=BLUE, linewidth=1.7, label="True state")
        ax.plot(t, filtered[sample, :, component], color=ORANGE, linewidth=1.3,
                linestyle="--", label="Corrected estimate")
        ax.axvspan(0, 0.8, color=GRAY, alpha=0.08)
        ax.set_ylabel(labels[component])
        ax.grid(axis="y", linewidth=0.7)
        ax.set_xlim(t[0], t[-1])
    axes[0].legend(loc="upper right", ncol=3, fontsize=9, frameon=False)
    axes[-1].set_xlabel("Time (Lorenz units); shaded region is initial convergence")
    fig.suptitle("State estimation on a held-out trajectory", fontsize=17, fontweight="bold")
    save(fig, "state_estimation")

    origin = int(data["origin"])
    origin_index = list(data["evaluation_origins"]).index(origin)
    selected_forecast = data["forecasts"][sample, origin_index]
    forecast_time = np.arange(len(selected_forecast) + 1) * config.dt
    forecast_with_origin = np.concatenate([filtered[sample, origin:origin + 1], selected_forecast])
    past_time = (np.arange(origin + 1) - origin) * config.dt
    all_time = (np.arange(origin + len(selected_forecast) + 1) - origin) * config.dt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for component, ax in enumerate(axes):
        ax.plot(all_time, truth[sample, :len(all_time), component], color=BLUE,
                linewidth=1.8, label="True state (evaluation only)")
        ax.plot(past_time, filtered[sample, :origin + 1, component], color=OLIVE,
                linewidth=1.5, linestyle=":", label="History estimate")
        ax.plot(forecast_time, forecast_with_origin[:, component], color=ORANGE,
                linewidth=1.7, linestyle="--", label="Measurement-free forecast")
        if component < sensors:
            ax.scatter(past_time[::5], measurements[sample, :origin + 1:5, component],
                       color=GRAY, alpha=0.5, s=10)
        ax.axvline(0, color="#394254", linewidth=1)
        ax.axvspan(0, config.horizon * config.dt, color=ORANGE, alpha=0.045)
        ax.set_ylabel(labels[component])
        ax.grid(axis="y", linewidth=0.7)
        ax.set_xlim(past_time[0], forecast_time[-1])
    axes[0].legend(loc="upper right", ncol=3, fontsize=9, frameon=False)
    axes[-1].set_xlabel("Time from last assimilated measurement (Lorenz units)")
    fig.suptitle("Forecast from time zero — no future measurements enter the model",
                 fontsize=16, fontweight="bold")
    save(fig, "forecast")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    steps = [row["step"] for row in log]
    axes[0].semilogy(steps, [row["train_future"] for row in log], color=BLUE,
                     linewidth=1.8, label="Training (fixed 64-sequence subset)")
    axes[0].semilogy(steps, [row["val_future"] for row in log], color=ORANGE,
                     linestyle="--", linewidth=1.8, label="Validation (64 independent sequences)")
    axes[0].axvline(metrics["best_validation_update"], color=GRAY, linestyle=":", linewidth=1)
    axes[0].set(xlabel="Adam update", ylabel="Future measurement MSE (log scale)",
                title=training_title, xlim=(0, steps[-1]))
    axes[0].grid(axis="y", which="major", linewidth=0.7)
    axes[0].legend(fontsize=8.5, frameon=False)
    names = ["u → du/dt", "v → du/dt", "u → dv/dt", "v → dv/dt", "w → dw/dt"]
    locations = np.arange(len(names))
    actual = np.array(metrics["true_coefficients"])
    learned = np.array(metrics["learned_coefficients"])
    axes[1].barh(locations - 0.17, actual, height=0.32, color=BLUE, label="True coefficient")
    axes[1].barh(locations + 0.17, learned, height=0.32, facecolor="white",
                 edgecolor=ORANGE, linewidth=1.8, label="Learned coefficient")
    axes[1].axvline(0, color=GRAY, linewidth=0.8)
    axes[1].set(yticks=locations, yticklabels=names, xlabel="Signed coefficient",
                title="Dynamics within the chosen Lorenz term family")
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    for loc, value in zip(locations, learned):
        axes[1].text(value + (0.5 if value >= 0 else -0.5), loc + 0.17,
                     f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    axes[1].set_xlim(-15, 35)
    save(fig, "training_and_coefficients")

    fig, ax = plt.subplots(figsize=(11, 5.3), layout="constrained")
    lead = np.arange(1, len(data["lead_rmse"]) + 1) * config.dt
    ax.plot(lead, data["persistence_rmse"], color=GRAY, linewidth=1.7, linestyle=":",
            label="Persistence: repeat last noisy measurement")
    ax.plot(lead, data["fixed_lead_rmse"], color=OLIVE, linewidth=1.7, linestyle="--",
            label="Learned dynamics + fixed corrector only")
    ax.plot(lead, data["lead_rmse"], color=BLUE, linewidth=2,
            label="Learned dynamics + learned corrector")
    ax.axvline(config.horizon * config.dt, color=ORANGE, linewidth=1.2, linestyle="--",
               label=f"Training horizon ({config.horizon * config.dt:g} units)")
    ax.set(xlabel="Forecast lead time (Lorenz units)", ylabel="Sensor RMSE against noise-free truth",
           title=f"Held-out forecast error: {config.test_sequences} sequences × 4 origins",
           xlim=(0, lead[-1]), ylim=(0, None))
    ax.grid(axis="y", linewidth=0.7)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save(fig, "forecast_skill")
