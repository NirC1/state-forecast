"""Tests of causal forecasts and the complete gradient path."""
import torch

from examples.lorenz import Config, make_model


def test_zero_initialized_corrector_is_pseudoinverse_correction():
    model = make_model(Config())
    prior = torch.tensor([[2., -3., 5.]])
    measurement = torch.tensor([[-1., 4.]])
    posterior = model.correct(prior, measurement)
    torch.testing.assert_close(posterior, torch.tensor([[-1., 4., 5.]]))
    assert torch.count_nonzero(model.corrector[-1].weight) == 0


def test_future_measurements_do_not_change_an_origin_or_its_forecast():
    torch.manual_seed(2)
    model = make_model(Config())
    measurements = torch.randn(2, 12, 2)
    changed = measurements.clone()
    changed[:, 7:] += 50
    first_origin = model.filter(measurements[:, :7])[:, -1]
    second_origin = model.filter(changed[:, :7])[:, -1]
    torch.testing.assert_close(model.forecast(first_origin, 5), model.forecast(second_origin, 5))
    assert not torch.allclose(model.loss(measurements, (6,), 5)[0],
                              model.loss(changed, (6,), 5)[0])


def test_future_loss_reaches_early_corrections_and_dynamics():
    torch.manual_seed(4)
    model = make_model(Config())
    measurements = torch.randn(2, 10, 2)
    first_posterior = model.correct(torch.zeros(2, 3), measurements[:, 0])
    first_posterior.retain_grad()
    state = first_posterior
    for t in range(1, 6):
        state = model.correct(model.dynamics(state), measurements[:, t])
    loss = (model.observe(model.forecast(state, 4)) - measurements[:, 6:]).square().mean()
    loss.backward()
    assert first_posterior.grad is not None and first_posterior.grad.abs().sum() > 0
    assert model.dynamics.coefficients.grad.abs().sum() > 0
    assert model.corrector[-1].weight.grad.abs().sum() > 0


def test_multiple_origins_match_independent_prefixes_and_current_count_once():
    torch.manual_seed(5)
    model = make_model(Config())
    measurements = torch.randn(3, 14, 2)
    origins, horizon = (5, 8), 5
    total, future, current = model.loss(measurements, origins, horizon)
    independent = []
    for origin in origins:
        state = model.filter(measurements[:, :origin + 1])[:, -1]
        predicted = model.observe(model.forecast(state, horizon))
        independent.append((predicted - measurements[:, origin + 1:origin + 1 + horizon]).square().mean())
    torch.testing.assert_close(future, torch.stack(independent).mean())
    history = measurements[:, :max(origins) + 1]
    torch.testing.assert_close(current, (model.observe(model.filter(history)) - history).square().mean())
    torch.testing.assert_close(total, future + 0.1 * current)


def test_generic_observation_operator_and_frozen_forecast():
    h = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    model = make_model(Config(), observation=h)
    y = torch.tensor([[[2., 4., -1.]]])
    posterior = model.filter(y)
    torch.testing.assert_close(model.observe(posterior), y)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        frozen_posterior = model.filter(y)
        assert not model.forecast(frozen_posterior[:, -1], 3).requires_grad


def test_each_independent_sequence_is_reset_to_zero():
    torch.manual_seed(8)
    model = make_model(Config())
    measurements = torch.randn(2, 6, 2)
    batched = model.filter(measurements)
    separate = torch.cat([model.filter(measurements[i:i + 1]) for i in range(2)], 0)
    torch.testing.assert_close(batched, separate)


def test_online_assimilation_equals_complete_history_filter():
    torch.manual_seed(10)
    model = make_model(Config())
    measurements = torch.randn(2, 8, 2)
    posterior = None
    for measurement in measurements.unbind(1):
        posterior = model.assimilate(measurement, posterior)
    torch.testing.assert_close(posterior, model.filter(measurements)[:, -1])


def test_three_sensor_loss_is_connected_and_averages_all_sensors():
    torch.manual_seed(11)
    model = make_model(Config(sensors=3))
    measurements = torch.randn(2, 9, 3)
    loss, future, current = model.loss(measurements, (4,), 4)
    torch.testing.assert_close(model.H, torch.eye(3))
    torch.testing.assert_close(current, torch.tensor(0.), atol=1e-6, rtol=0.)
    state = model.filter(measurements[:, :5])[:, -1]
    expected = (model.forecast(state, 4) - measurements[:, 5:9]).square().mean()
    torch.testing.assert_close(future, expected)
    loss.backward()
    assert model.dynamics.coefficients.grad.abs().sum() > 0
