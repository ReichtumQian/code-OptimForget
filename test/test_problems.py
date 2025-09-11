import torch
import pytest
from unittest.mock import MagicMock
from torch.autograd.functional import hvp

# Adjust the import path as necessary
from optforget import (OptimControlProblem, PMPProblem, FineTuneForgetProblem)

# ==================================================================
# 1. Tests for OptimControlProblem (Base Class)
# ==================================================================


def test_optim_control_problem_initialization(base_problem_components):
    """
    Tests if the OptimControlProblem class correctly initializes its attributes.
    """
    problem = OptimControlProblem(**base_problem_components)

    # Check that all components are correctly assigned
    assert problem.f is base_problem_components['f']
    assert problem.running_cost is base_problem_components['running_cost']
    assert problem.terminal_cost is base_problem_components['terminal_cost']
    assert problem.t0 == 0.0
    assert problem.tf == 1.0

    # Check if dimensions are inferred correctly
    assert problem.batch_size == 1
    assert problem.state_dim == 2
    assert problem.control_dim == 2
    assert torch.equal(problem.x0, base_problem_components['x0'])


def test_optim_control_problem_x0_unsqueezing():
    """
    Tests if a 1D initial state `x0` is correctly unsqueezed to have a batch dimension.
    """
    # Create a simple mock for the callable functions
    mock_func = MagicMock(return_value=torch.tensor(0.0))
    x0_1d = torch.zeros(5)  # A 1D tensor

    problem = OptimControlProblem(mock_func, mock_func, mock_func, x0_1d, 0.0,
                                  1.0, 5)

    # Check if x0 now has shape (1, 5)
    assert problem.x0.shape == (1, 5)
    assert problem.batch_size == 1
    assert problem.state_dim == 5


def test_evaluate_cost_calculation(base_problem_components):
    """
    Tests the `evaluate_cost` method with a known analytical solution.
    For L=0.5*u^2, f=u, x0=0, the cost is J = 2*||x_f - 1||^2 + integral(0.5*u^2)dt.
    If we choose a constant control u=[1, 1], then x_f = x0 + integral(u)dt = [1, 1].
    The terminal cost will be 0.
    The integral cost will be integral(0.5 * (1^2 + 1^2))dt = integral(1)dt = 1.0.
    Total cost should be 1.0.
    """
    problem = OptimControlProblem(**base_problem_components)

    K = 11  # 11 time steps
    t_arr = torch.linspace(0.0, 1.0, K)

    # State trajectory: x(t) = t * [1, 1]. x_traj shape: (K, N, state_dim)
    x_traj = torch.ones(K, 1, 2) * t_arr.view(K, 1, 1)

    # Control trajectory: u(t) = [1, 1] constant. u_traj shape: (K-1, N, control_dim)
    u_traj = torch.ones(K - 1, 1, 2)

    total_cost = problem.evaluate_cost(t_arr, x_traj, u_traj)

    # The expected integral cost is 1.0. The terminal cost is 0.
    assert abs(total_cost - 1.0) < 1e-6


# ==================================================================
# 2. Tests for PMPProblem (Extended Class)
# ==================================================================


class ConcretePMPProblem(PMPProblem):
    """A minimal concrete implementation of PMPProblem for testing purposes."""

    def compute_terminal_costate(self, t_f, x_f):
        return torch.ones_like(x_f)

    def compute_costate_dynamics(self, t, x, p, u):
        return torch.zeros_like(p)

    def compute_optimal_control(self, t, x, p):
        return -p


def test_pmp_hamiltonian_calculation(base_problem_components):
    """
    Tests the Hamiltonian calculation H = L + p^T * f.
    """
    problem = ConcretePMPProblem(**base_problem_components)

    t, x, p, u = 0.5, torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2)

    # Manually calculate H
    L = base_problem_components['running_cost'](t, x, u)  # 0.5 * ||u||^2
    f_val = base_problem_components['f'](t, x, u)  # u
    pT_f = torch.sum(p * f_val, dim=1)
    expected_H = L + pT_f

    # Calculate H using the method
    actual_H = problem.hamiltonian(t, x, p, u)

    assert actual_H.shape == (1, )
    assert torch.allclose(actual_H, expected_H)


def test_pmp_abstract_methods_raise_error(base_problem_components):
    """
    Tests that the abstract methods in the base PMPProblem raise NotImplementedError.
    """
    problem = PMPProblem(**base_problem_components)
    dummy_tensor = torch.zeros(1, 2)

    with pytest.raises(NotImplementedError):
        problem.compute_terminal_costate(1.0, dummy_tensor)
    with pytest.raises(NotImplementedError):
        problem.compute_costate_dynamics(0.0, dummy_tensor, dummy_tensor,
                                         dummy_tensor)
    with pytest.raises(NotImplementedError):
        problem.compute_optimal_control(0.0, dummy_tensor, dummy_tensor)


# ==================================================================
# 3. Tests for FineTuneForgetProblem (Concrete Class)
# ==================================================================


def test_fine_tune_problem_compute_loss_l2_raises_error(fine_tune_problem):
    """
    Tests that calling _compute_loss_l2 without setting a batch raises a ValueError.
    """
    x = fine_tune_problem.x0
    with pytest.raises(ValueError, match="Current batch has not been set"):
        fine_tune_problem._compute_loss_l2(x)


def test_fine_tune_problem_compute_loss_l2_correctness(fine_tune_problem,
                                                       simple_model,
                                                       mock_dataloader,
                                                       loss_function):
    """
    Compares the loss from functional_call with a standard model forward pass.
    """
    # Get a batch of data and set it in the problem
    batch = next(iter(mock_dataloader))
    fine_tune_problem.set_current_batch(batch)

    # 1. Calculate loss using the problem's method
    x = fine_tune_problem.x0
    loss_from_problem = fine_tune_problem._compute_loss_l2(x)

    # 2. Calculate loss using a standard model pass
    inputs, labels = batch
    outputs = simple_model(inputs)
    expected_loss = loss_function(outputs, labels)

    assert torch.allclose(loss_from_problem, expected_loss)


def test_fine_tune_problem_costate_dynamics_hvp(fine_tune_problem,
                                                mock_dataloader):
    """
    Tests the costate dynamics by comparing it against PyTorch's `hvp` function.
    This is the most critical test for this class.
    dp/dt = eta * HVP(grad_L2, p)
    """
    # Setup: get a batch and some dummy tensors
    batch = next(iter(mock_dataloader))
    fine_tune_problem.set_current_batch(batch)

    x = fine_tune_problem.x_anchor.clone().requires_grad_(True)
    p = torch.randn_like(x)
    u = torch.randn_like(
        x)  # u is not used in dp/dt but required by the function signature

    # 1. Calculate dp/dt using the problem's method
    dp_dt_actual = fine_tune_problem.compute_costate_dynamics(
        0.0, x.unsqueeze(0), p.unsqueeze(0), u.unsqueeze(0))

    # 2. Calculate dp/dt using the reference `hvp` function
    def loss_fn_for_hvp(params):
        # The functional_call inside _compute_loss_l2 is hard to pass directly.
        # So we re-implement the loss calculation here for the HVP test.
        # This is a common pattern when testing complex autograd functions.
        params_list = params.split(fine_tune_problem.param_numels)
        param_dict = {
            name: p.view(shape)
            for name, p, shape in
            zip(fine_tune_problem.param_names, params_list,
                fine_tune_problem.param_shapes)
        }
        inputs, labels = batch
        predictions = torch.func.functional_call(fine_tune_problem.model,
                                                 param_dict, (inputs, ))
        return fine_tune_problem.loss_fn(predictions, labels)

    # The loss function for HVP needs to be a function of the parameter vector
    _, (hvp_result,) = hvp(loss_fn_for_hvp, (x, ), (p, ))
    dp_dt_expected = fine_tune_problem.eta * hvp_result

    assert dp_dt_actual.shape == dp_dt_expected.unsqueeze(0).shape
    assert torch.allclose(dp_dt_actual.squeeze(), dp_dt_expected, atol=1e-6)


def test_fine_tune_problem_optimal_control(fine_tune_problem):
    """
    Tests the simple analytical formula for the optimal control u* = -p / lambda.
    """
    p = torch.randn(1, fine_tune_problem.control_dim)

    # Calculate u* using the method
    u_star = fine_tune_problem.compute_optimal_control(0.0, None, p)

    # Manually calculate the expected result
    expected_u = -p / fine_tune_problem.lambda_reg

    assert u_star.shape == expected_u.shape
    assert torch.allclose(u_star, expected_u)


def test_fine_tune_problem_terminal_costate(fine_tune_problem,
                                            mock_dataloader):
    """
    Tests the gradient calculation for the terminal costate p(tf).
    p(tf) = d(phi)/dx = c1 * (x_f - x_anchor) + c2 * grad(L2)
    """
    # Setup
    batch = next(iter(mock_dataloader))
    fine_tune_problem.set_current_batch(batch)

    x_f = fine_tune_problem.x_anchor.clone().requires_grad_(True)

    # 1. Calculate p(tf) using the problem's method
    p_tf_actual = fine_tune_problem.compute_terminal_costate(
        1.0, x_f.unsqueeze(0))

    # 2. Calculate p(tf) manually using autograd
    # We need to re-create the terminal cost function here to get the gradient
    loss_l2 = fine_tune_problem._compute_loss_l2(x_f.unsqueeze(0))
    loss_l1 = 0.5 * torch.sum((x_f - fine_tune_problem.x_anchor)**2)
    phi = fine_tune_problem.c1 * loss_l1 + fine_tune_problem.c2 * loss_l2
    p_tf_expected = torch.autograd.grad(phi, x_f)[0]

    assert p_tf_actual.shape == p_tf_expected.unsqueeze(0).shape
    assert torch.allclose(p_tf_actual.squeeze(), p_tf_expected)
