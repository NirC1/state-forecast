"""Exercise current-format checkpoints using fresh measurement-only training."""
import json

import numpy as np
import pytest
import torch

from examples.lorenz import Config, evaluate, load_checkpoint, make_model, simulate, train


@pytest.mark.parametrize("sensors", [1, 2, 3])
def test_checkpoint_round_trip_and_warm_start(tmp_path, sensors):
    config = Config(sensors=sensors, train_sequences=4, val_sequences=2,
                    batch_size=2, origins=(3,), horizon=2, updates=1,
                    validate_every=1)
    model, initial, history, best_step = train(config, tmp_path / "initial")
    restored, restored_config, saved = load_checkpoint(tmp_path / "initial/checkpoint.pt")
    _, readings = simulate(2, 8, config, seed=100)
    measurements = torch.from_numpy(readings)
    with torch.no_grad():
        states = model.filter(measurements)
        restored_states = restored.filter(measurements)
        torch.testing.assert_close(restored_states, states, rtol=0, atol=0)
        torch.testing.assert_close(restored.forecast(restored_states[:, -1], 3),
                                   model.forecast(states[:, -1], 3), rtol=0, atol=0)
    assert restored_config == config
    assert saved["best_step"] == best_step
    assert len(history) == 2
    assert all(not parameter.requires_grad for parameter in restored.parameters())

    # With zero further updates, continuation must retain the selected model
    # and preserve the original pre-training state for evaluation.
    config.updates = 0
    continued, original, _, _ = train(config, tmp_path / "continued",
                                       tmp_path / "initial/checkpoint.pt")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(continued.state_dict()[key], value, rtol=0, atol=0)
    for key, value in initial.items():
        torch.testing.assert_close(original[key], value, rtol=0, atol=0)


def test_one_sensor_evaluation_saves_consistent_arrays_and_metrics(tmp_path):
    config = Config(sensors=1, test_sequences=2)
    model = make_model(config).eval().requires_grad_(False)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    metrics = evaluate(model, initial, config, tmp_path)

    assert metrics["sensors"] == 1
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    with np.load(tmp_path / "evaluation.npz") as arrays:
        assert arrays["measurements"].shape == (2, 1001, 1)
        assert arrays["filtered"].shape == (2, 1001, 3)
        assert arrays["initial_filtered"].shape == (2, 101, 3)
        assert arrays["forecasts"].shape == (2, 4, 200, 3)
        rmse = np.sqrt(np.mean((arrays["forecasts"][:, :, :40, :1]
                               - arrays["forecast_truth"][:, :, :40, :1]) ** 2))
        assert np.isfinite(arrays["forecasts"]).all()
        np.testing.assert_allclose(rmse, metrics["forecast_sensor_rmse_first_0.4_time_units"])
    assert (tmp_path / "forecast_errors.csv").is_file()


def test_checkpoint_with_legacy_current_weight_can_be_loaded(tmp_path):
    from dataclasses import asdict

    config = Config()
    model = make_model(config)
    legacy_config = {**asdict(config), "current_weight": .1}
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "config": legacy_config}, checkpoint)
    restored, restored_config, _ = load_checkpoint(checkpoint)
    assert restored_config == config
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
