"""Compare matched Lorenz test trajectories from the two sensor configurations."""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    partial_dir, full_dir = Path("results"), Path("results_full_state")
    partial = np.load(partial_dir / "evaluation.npz")
    full = np.load(full_dir / "evaluation.npz")
    np.testing.assert_array_equal(partial["truth"], full["truth"])
    partial_metrics = json.loads((partial_dir / "metrics.json").read_text())
    full_metrics = json.loads((full_dir / "metrics.json").read_text())
    fig = plt.figure(figsize=(15, 5.5), layout="constrained")
    views = [(partial["truth"][0, 80:], "#2563eb", "True trajectory"),
             (partial["filtered"][0, 80:], "#c87516", "Two sensors: u, v (w hidden)"),
             (full["filtered"][0, 80:], "#707b32", "Three sensors: u, v, w")]
    for index, (states, color, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.plot(*states.T, color=color, linewidth=0.85)
        ax.set(xlabel="u", ylabel="v", xlim=(-22, 22),
               ylim=(-30, 30), zlim=(0, 52), title=title)
        ax.text2D(0.96, 0.57, "w", transform=ax.transAxes, ha="right", fontsize=11)
        ax.view_init(elev=24, azim=-63)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
    fig.suptitle("Lorenz reconstruction on the same held-out trajectory", fontsize=17, fontweight="bold")
    fig.savefig("sensor_comparison.png", dpi=160, bbox_inches="tight")
    fig.savefig("sensor_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    summary = {}
    for name, metrics in [("two_sensors", partial_metrics), ("three_sensors", full_metrics)]:
        summary[name] = {key: metrics[key] for key in [
            "sensors", "state_rmse_after_0.8_time_units",
            "forecast_sensor_rmse_first_0.4_time_units",
            "persistence_sensor_rmse_first_0.4_time_units",
            "learned_coefficients", "coefficient_relative_error_percent"]}
    Path("comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
