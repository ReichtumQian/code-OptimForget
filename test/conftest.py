import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.convert_parameters import parameters_to_vector

from optforget import (OptimControlProblem, PMPProblem, FineTuneForgetProblem)


@pytest.fixture
def simple_model():
    """Provides a simple, deterministic linear model for testing."""
    model = nn.Linear(10, 2, bias=False)
    # Use a fixed weight initialization for reproducibility
    torch.manual_seed(42)
    with torch.no_grad():
        model.weight.fill_(0.1)
    return model


@pytest.fixture
def initial_params_vector(simple_model):
    """Provides the initial parameters of the simple_model as a flat vector."""
    # return size([20])
    return parameters_to_vector(simple_model.parameters()).detach().clone()


@pytest.fixture
def mock_dataloader():
    """Provides a simple, small DataLoader."""
    inputs = torch.randn(16, 10)
    labels = torch.randint(0, 2, (16, ))
    dataset = TensorDataset(inputs, labels)
    # Use a small batch size for quick tests
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def loss_function():
    """Provides a standard loss function."""
    return nn.CrossEntropyLoss()


# --- Fixtures for Problem Classes ---


@pytest.fixture
def base_problem_components():
    """
    Provides a set of simple, analytically solvable components
    for an OptimControlProblem.
    """

    # System dynamics: dx/dt = u. A simple integrator.
    def f_dynamics(t, x, u):
        return u

    # Running cost: L = 0.5 * ||u||^2
    def running_cost(t, x, u):
        return 0.5 * torch.sum(u * u, dim=1)

    # Terminal cost: phi = 2.0 * ||x_f - x_target||^2
    def terminal_cost(t, x):
        x_target = torch.ones_like(x)
        return 2.0 * torch.sum((x - x_target)**2, dim=1)

    return {
        'f': f_dynamics,
        'running_cost': running_cost,
        'terminal_cost': terminal_cost,
        'x0': torch.zeros(1, 2),  # batch_size=1, state_dim=2
        't0': 0.0,
        'tf': 1.0,
        'control_dim': 2
    }


@pytest.fixture
def fine_tune_problem(simple_model, initial_params_vector, mock_dataloader,
                      loss_function):
    """
    Provides a fully initialized FineTuneForgetProblem instance for testing.
    """
    problem_params = {
        'lambda_reg': 0.1,
        'c_costs': (1.0, 5.0),
        'eta': 0.01,
        'ft_dataloader': mock_dataloader,
        'model': simple_model,
        'loss_function': loss_function,
        'x_anchor': initial_params_vector,
        't0': 0.0,
        'tf': 1.0,
    }
    return FineTuneForgetProblem(**problem_params)
