# Joint state estimation and dynamics discovery on Lorenz

The reusable learning implementation lives in `state_forecast/`. Open
`notebooks/lorenz.ipynb` for a training walkthrough. Run the commands below from
the repository root. All result folders and comparison files are generated locally
and excluded from version control.

This experiment implements the supplied advance–correct–forecast training loop in
PyTorch, with one, two, or three noisy sensors and a three-dimensional hidden
state. It learns dynamics coefficients and a neural corrector jointly from
measurements only. The main notebook defaults to one sensor (`u`) and retains its
executed outputs. The command-line runner defaults to two sensors (`u,v`);
three sensors measure all coordinates with noise to assess recovery when
hidden-coordinate ambiguity is reduced.

## Run

Use Python 3.11 or newer. CPU training is sufficient.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[examples,test]"
.venv/bin/python -m examples.lorenz --updates 1600 --test-seed 10 --output results_short_horizon
.venv/bin/python -m examples.lorenz --warm-start results_short_horizon/checkpoint.pt --horizon 100 --updates 1200 --corrector-lr 0.0005 --output results
.venv/bin/python -m examples.lorenz --sensors 3 --updates 800 --corrector-lr 0.0005 --output results_full_state_initial
.venv/bin/python -m examples.lorenz --sensors 3 --warm-start results_full_state_initial/checkpoint.pt --updates 800 --dynamics-lr 0.02 --corrector-lr 0.00025 --output results_full_state
.venv/bin/python -m examples.compare_lorenz
.venv/bin/python -m pytest -q
```

The reference experiment used Python 3.14 and PyTorch 2.9.1 CPU. The initial 0.4-unit horizon
run is in `results_short_horizon/`; the longer-horizon refinement is in `results/`.
The three-sensor experiment starts from scratch in `results_full_state_initial/`
and receives 800 additional lower-rate updates in `results_full_state/`.
The refinement starts from the best first-phase checkpoint and creates a fresh
Adam optimizer. Its training curve counts refinement updates starting at zero.
Both displayed final learning curves show their refinement phase; the initial
phase curves are retained in the corresponding initial results folders.
To regenerate evaluation and figures from the saved best-validation checkpoint:

```bash
.venv/bin/python -m examples.lorenz --evaluate-only
```

Checkpoints use the current library state dictionary format. Reproduce the
commands above to generate them before using `--evaluate-only` or the online example.

Use `--updates 100 --output smoke` for a short execution check, or
`--integrator euler --output results_euler` for the explicit Euler construction.
Use `--sensors 1` to measure only `u`, leaving both `v` and `w` unmeasured:

```bash
.venv/bin/python -m examples.lorenz --sensors 1 --updates 800 --corrector-lr 0.0005 --output results_one_sensor_initial
.venv/bin/python -m examples.lorenz --sensors 1 --warm-start results_one_sensor_initial/checkpoint.pt --updates 800 --dynamics-lr 0.02 --corrector-lr 0.00025 --output results_one_sensor
```

Euler at a finite sampling interval can bias the inferred continuous-time
coefficients; the default uses differentiable RK4 inside the one-step map Phi.

## Model and assumptions

Write the physical state as `(u, v, w)` to distinguish it from sensor readings `y`.
The simulator uses the standard Lorenz equations:

```
du/dt = 10 (v - u)
dv/dt = 28 u - v - u w
dw/dt = u v - (8/3) w
```

The discovered family is:

```
du/dt = a u + b v
dv/dt = c u + d v - u w
dw/dt = u v + e w
```

All five linear coefficients `(a,b,c,d,e)` are unconstrained signed parameters,
initialized at `(-6,6,18,-0.5,-1.5)`. The fixed quadratic coefficients are a
structural prior: this is discovery within a supplied Lorenz term family, not
blind discovery from a general polynomial library. Fixing `-u*w` and `+u*v`
anchors the physical scale of the unobserved coordinate. Letting both interaction
coefficients float with only `(u,v)` observed can admit hidden-state rescalings;
good measurement forecasts alone would then not establish a unique physical `w`.

Measurements are `y = Hx + noise`, with fixed
`H = [[1,0,0],[0,1,0]]`. Gaussian noise has standard deviation 0.3 state units.
`A = pinv(H)` is computed once and saved as a model buffer. The learned correction
uses a 4→64→3 network with Tanh and a zero-initialized final layer. Fixed input and
output scaling condition this network; losses use unnormalized physical units.
The correction is `x_plus = x_minus + A e + C([e,y])`. No positivity clamps are used.
For `--sensors 3`, `H` is the identity and the network has 6 inputs. The true state
is still hidden behind noisy readings, but no coordinate is wholly unmeasured.
For `--sensors 1`, `H = [[1,0,0]]` and the correction network has 2 inputs.

## Training and causality

- Sampling interval: 0.01 Lorenz time units.
- 256 training, 64 validation, and 64 test trajectories have independently drawn
  initial conditions, separate simulation seeds, and independent sensor noise.
  Each trajectory has a 3-unit simulator burn-in before its first measurement.
- Each independent history starts from a zero estimate at its first measurement.
  The filter processes all supplied measurements, without detaching states.
- Each training sequence has forecast origins at sample indices 60, 80, and 100,
  each with 40 future targets (0.4 time units) initially and 100 future targets
  (1.0 time units) in refinement. An origin at index 100 has a
  complete local history `y_-100,...,y_0` after shifting its time to zero.
- Shared history is processed once through index 100. Earlier saved states are
  causal: corrections at later indices cannot alter their forecasts. Current
  errors are counted once per history time, and future errors are averaged over
  sequence, origin, lead, and sensor.
- The exact objective is `future_MSE + 0.1 * current_MSE`.
  Adam jointly updates dynamics and correction parameters. Gradient norm clipping
  at 20 limits instability; there is no state detachment or truncated gradient.
- Training and validation true states are discarded. Validation measurement loss
  selects the checkpoint. Test truth is used only for state/forecast scoring and
  comparisons with known coefficients.
- The final test uses a fresh simulation/noise seed (101), separate from the
  initial run's test seed (10). The horizon and optimizer refinement responds to
  first-phase loss oscillation and incomplete coefficient convergence.
- In evaluation parameters are frozen, the filter continues online, and forecasts
  use copied corrected states with the learned dynamics alone.

Test metrics use all 64 test sequences. Forecast-error curves aggregate four
origins per sequence (80,100,120,140) and extend to 2 time units, beyond the 1.0-unit
refinement horizon. These origins share trajectories; they are not 256 independent
trajectories. State errors exclude the first 0.8 units of estimator convergence.
Displayed sample paths always use test sequence 0; they are not chosen by accuracy.

To use the trained model with arriving measurements:

```python
import torch
from pathlib import Path
from examples.lorenz import load_checkpoint

model, config, saved = load_checkpoint(Path("results/checkpoint.pt"))

posterior = None  # Set to None only when starting an independent sequence.
with torch.no_grad():
    for sensor_reading in arriving_measurements:  # A tensor [u_sensor, v_sensor].
        posterior = model.assimilate(sensor_reading, posterior)
        future_states = model.forecast(posterior.clone(), horizon=40)
        future_measurements = model.observe(future_states)
```

The fixed-corrector comparison uses the same jointly learned dynamics and disables
the neural correction. It is an ablation, not a separately optimized baseline.
Persistence repeats the latest noisy sensor readings. Error curves are aggregate
RMSE values, not confidence intervals. This is a single-seed synthetic experiment.
The two- and three-sensor RMSE curves average different sensor sets; compare their
individual baseline improvements rather than treating the curves as identical
metrics. The two-sensor run retains substantial hidden-coordinate bias; good
sensor forecasts do not by themselves establish accurate state recovery.

## Graphs and artifacts

- `sensor_comparison.png`: true, two-sensor, and three-sensor attractors on the
  same test trajectory. `comparison.json` records aggregate numerical results.
- `results/attractor.png`: true and reconstructed trajectories in 3D.
- `results/state_estimation.png`: all three estimated coordinates, including hidden w.
- `results/forecast.png`: history and a forecast from time zero without future input.
- `results/training_and_coefficients.png`: learning curves and signed coefficients.
- `results/forecast_skill.png`: held-out errors versus forecast lead and baselines.
- Each figure also has an SVG version.
- The same figure types are saved in `results_full_state/` for the three-sensor run.
- `results/checkpoint.pt`: frozen best-validation parameters and initial parameters.
- `results/config.json`, `training.json`, `metrics.json`: settings, learning trace,
  and evaluation results.
- `results/evaluation.npz`, `forecast_errors.csv`: numerical evidence behind figures.

## Reference results

These historical single-seed results provide context; generated checkpoints and
figures are not distributed with the source. Exact results may vary by environment.

RMSE is in nondimensional Lorenz state units. Forecast numbers below average
the first 0.4 time units over 64 test sequences and four origins per sequence.
Each forecast comparison uses the sensors available in that experiment.

| Run | w state RMSE | Forecast sensor RMSE | Persistence RMSE |
| --- | ---: | ---: | ---: |
| Two sensors; w unmeasured | 4.421 | 0.409 | 7.521 |
| Three sensors; w measured with noise | 0.284 | 0.370 | 8.955 |

The final three-sensor coefficients are `(-9.919, 9.873, 27.773, -0.786, -2.6665)`.
Four are within 1.3% of their known simulator values; the `v` coefficient in
`dv/dt` has 21.4% relative error. The two-sensor fit has more substantial coefficient
and hidden-state bias. The joint objective produces useful short-term forecasts
without guaranteeing exact physical state or coefficient recovery.

Forecast error grows beyond the training horizon. The three-sensor aggregate
sensor RMSE is approximately 0.675 at lead 0.4, 3.325 at lead 1.0, and 5.704
at lead 2.0. These are errors at specific leads, distinct from horizon-averaged
errors in the table. No uncertainty intervals or multi-seed robustness claim
are provided by this single-seed experiment.

Tests check the zero-initialized correction, forecast causality, gradient flow to
early corrections, multi-origin averaging, online updates, and one-, two-, and
three-sensor observation matrices.

The Lorenz equations and their sensitivity to initial conditions originate in
[Lorenz (1963), Deterministic Nonperiodic Flow](https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2).
