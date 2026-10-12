"""Exercise the library with systems other than Lorenz and arbitrary state sizes."""
import itertools

import pytest
import torch
from torch import nn

from state_forecast import JointEstimator, ODEDynamics, TrainingConfig, fit


class ScalarDynamics(nn.Module):
    def __init__(self, initial=0.5):
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(initial))

    def forward(self, state):
        return self.factor * state


def test_four_component_state_and_nontrivial_two_sensor_observation():
    h = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 2.]], dtype=torch.float64)
    model = JointEstimator(ScalarDynamics(), h, hidden_width=8)
    measurement = torch.tensor([[[2., -3.], [1., 4.]]], dtype=torch.float64)
    states = model.filter(measurement)
    assert states.shape == (1, 2, 4)
    assert states.dtype == torch.float64
    torch.testing.assert_close(model.observe(states), measurement)
    assert model.forecast(states[:, -1], 3).shape == (1, 3, 4)


def test_optional_physical_projection_applies_after_correction():
    model = JointEstimator(ScalarDynamics(), torch.eye(2), state_projection=lambda x: x.clamp_min(0))
    posterior = model.assimilate(torch.tensor([-2., 3.]))
    torch.testing.assert_close(posterior, torch.tensor([0., 3.]))
    unrestricted = JointEstimator(ScalarDynamics(), torch.eye(2))
    torch.testing.assert_close(unrestricted.assimilate(torch.tensor([-2., 3.])), torch.tensor([-2., 3.]))


@pytest.mark.parametrize('integrator', ['euler', 'rk4'])
def test_generic_integrator_matches_linear_ode_step(integrator):
    rate = nn.Linear(4, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        rate.weight.copy_(torch.diag(torch.tensor([-1., -2., -3., -4.], dtype=torch.float64)))
    dynamics = ODEDynamics(rate, 0.1, integrator)
    state = torch.tensor([[1., -2., 3., -4.]], dtype=torch.float64)
    z = 0.1 * rate.weight.diag()
    multiplier = 1 + z if integrator == 'euler' else 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    torch.testing.assert_close(dynamics(state), state * multiplier)
    dynamics(state).square().mean().backward()
    assert rate.weight.grad.abs().sum() > 0


def test_measurement_only_training_learns_scalar_dynamics_and_restores_best():
    torch.manual_seed(12)
    initial = torch.linspace(0.5, 1.5, 24).reshape(-1, 1, 1)
    measurements = initial * torch.pow(0.9, torch.arange(12)).reshape(1, -1, 1)
    validation = measurements[:8] * 1.1
    model = JointEstimator(ScalarDynamics(), torch.ones(1, 1), hidden_width=8)
    progress = []
    result = fit(model, measurements, validation,
                 TrainingConfig(origins=(3, 5), horizon=4, updates=80, batch_size=12,
                                dynamics_lr=0.02, corrector_lr=0.002, validate_every=20),
                 on_validation=progress.append)
    assert result.best_validation_loss < 0.1 * result.history[0]['val_loss']
    assert len(progress) == 5
    assert all(row['val_loss'] == row['val_future'] for row in result.history)
    assert all(row['train_loss'] == row['train_future'] for row in result.history)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, result.best_state[name])
    with pytest.raises(ValueError, match='no trainable parameters'):
        fit(model, measurements, validation, TrainingConfig(origins=(3,), horizon=4, updates=1))


def test_denoising_with_true_dynamics_attains_forecast_bayes_risk():
    # Enumerate X and two independent sensor errors in {-1, +1}. The state
    # stays constant, and E[Y1 | Y0] = Y0/2. Penalizing current disagreement
    # would make the incorrect contraction cheaper than this true model.
    data = torch.tensor([[[x + e0], [x + e1]]
                         for x, e0, e1 in itertools.product((-1., 1.), repeat=3)])
    true_model = JointEstimator(ScalarDynamics(1.), torch.ones(1, 1), hidden_width=1)
    with torch.no_grad():
        true_model.corrector[0].weight.copy_(torch.tensor([[0., 1.]]))
        true_model.corrector[0].bias.zero_()
        true_model.corrector[-1].weight.fill_(-1. / torch.tanh(torch.tensor(2.)))
    wrong_model = JointEstimator(ScalarDynamics(.5), torch.ones(1, 1), hidden_width=1)
    true_loss, true_future, true_current = true_model.loss(data, (0,), 1)
    wrong_loss, _, wrong_current = wrong_model.loss(data, (0,), 1)
    torch.testing.assert_close(true_future, torch.tensor(1.5))
    torch.testing.assert_close(true_current, torch.tensor(.5))
    torch.testing.assert_close(wrong_current, torch.tensor(0.))
    torch.testing.assert_close(true_loss, true_future)
    torch.testing.assert_close(true_loss, wrong_loss)


def test_training_rejects_missing_targets_and_nonfinite_data():
    model = JointEstimator(ScalarDynamics(), torch.ones(1, 1), hidden_width=8)
    data = torch.ones(2, 8, 1)
    with pytest.raises(ValueError, match='complete targets'):
        fit(model, data, data, TrainingConfig(origins=(6,), horizon=3, updates=1))
    data[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        fit(model, data, data, TrainingConfig(origins=(3,), horizon=3, updates=1))

