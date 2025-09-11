import torch
import pytest
from unittest.mock import MagicMock, call

# Adjust the import path as necessary
from optforget import (
    MSASolver,
    StochasticMSASolver,
    FineTuneForgetProblem
)

# ==================================================================
# 1. Tests for MSASolver (Deterministic Solver)
# ==================================================================

@pytest.fixture
def mock_pmp_problem():
    """
    Creates a MagicMock object that simulates a PMPProblem.
    This allows us to control the outputs of its methods for predictable testing.
    """
    problem = MagicMock()
    problem.x0 = torch.zeros(1, 2)  # batch=1, state_dim=2
    problem.batch_size = 1
    problem.state_dim = 2
    problem.control_dim = 2
    problem.t0 = 0.0
    problem.tf = 1.0
    return problem


def test_msa_solver_forward_pass(mock_pmp_problem):
    """
    Tests the forward Euler integration in _forward_pass.
    We set the dynamics f(t, x, u) to be a constant vector [1.0, -0.5].
    With x0 = [0, 0], we expect x(t) = [t, -0.5*t].
    """
    # Configure the mock problem's dynamics function
    mock_pmp_problem.f.return_value = torch.tensor([[1.0, -0.5]])

    num_steps = 11
    solver = MSASolver(num_steps=num_steps, num_iterations=1, problem=mock_pmp_problem)
    
    # The control trajectory can be zeros, as our mock `f` ignores it
    u_traj = torch.zeros(num_steps - 1, 1, 2)
    
    x_traj = solver._forward_pass(u_traj)

    # Check final state: x(tf=1.0) should be close to [1.0, -0.5]
    expected_x_final = torch.tensor([[1.0, -0.5]])
    assert x_traj.shape == (num_steps, 1, 2)
    assert torch.allclose(x_traj[-1], expected_x_final, atol=1e-5)


def test_msa_solver_backward_pass(mock_pmp_problem):
    """
    Tests the backward Euler integration in _backward_pass.
    We set the costate dynamics dp/dt to be a constant vector [0.1, 0.2].
    And the terminal costate p(tf) to be [1.0, 1.0].
    The backward integration is p_k = p_{k+1} - dt * (dp/dt).
    So we expect p(t0) = p(tf) - (tf - t0) * [0.1, 0.2]
                     = [1.0, 1.0] - 1.0 * [0.1, 0.2] = [0.9, 0.8]
    """
    # Configure the mock problem's methods
    mock_pmp_problem.compute_terminal_costate.return_value = torch.tensor([[1.0, 1.0]])
    mock_pmp_problem.compute_costate_dynamics.return_value = torch.tensor([[0.1, 0.2]])

    num_steps = 11
    solver = MSASolver(num_steps=num_steps, num_iterations=1, problem=mock_pmp_problem)
    
    # Dummy trajectories required as input
    x_traj = torch.zeros(num_steps, 1, 2)
    u_traj = torch.zeros(num_steps - 1, 1, 2)

    p_traj = solver._backward_pass(x_traj, u_traj)
    
    # Check the initial costate p(t0)
    expected_p_initial = torch.tensor([[0.9, 0.8]])
    assert p_traj.shape == (num_steps, 1, 2)
    assert torch.allclose(p_traj[0], expected_p_initial, atol=1e-5)


def test_msa_solver_update_control(mock_pmp_problem):
    """
    Tests that _update_control calls the problem's compute_optimal_control
    for each time step correctly.
    """
    # Let the optimal control be a function of the costate, e.g., -p
    mock_pmp_problem.compute_optimal_control.side_effect = lambda t, x, p: -p

    num_steps = 5
    solver = MSASolver(num_steps=num_steps, num_iterations=1, problem=mock_pmp_problem)
    
    # Create some dummy trajectories
    x_traj = torch.randn(num_steps, 1, 2)
    # Let costate p grow linearly for easy checking
    p_traj = torch.arange(num_steps, dtype=torch.float32).view(num_steps, 1, 1) * torch.ones(num_steps, 1, 2)

    u_new_traj = solver._update_control(x_traj, p_traj)

    # We expect u_k = -p_{k+1}
    expected_u_new_traj = -p_traj[1:] # p_traj from k=1 to K-1

    assert u_new_traj.shape == (num_steps - 1, 1, 2)
    assert torch.allclose(u_new_traj, expected_u_new_traj)
    # Verify that the mock was called for each step from k=0 to K-2
    assert mock_pmp_problem.compute_optimal_control.call_count == num_steps - 1


# ==================================================================
# 2. Tests for StochasticMSASolver
# ==================================================================

def test_stochastic_solver_get_next_batch_resets(fine_tune_problem):
    """
    Tests that the data iterator resets after being exhausted.
    """
    # The dataloader in the fixture has 16 samples with batch_size=4, so 4 batches.
    solver = StochasticMSASolver(num_steps=10, num_iterations=1, problem=fine_tune_problem)
    
    # Exhaust the iterator
    for _ in range(4):
        solver.problem._get_next_batch()
    
    # The 5th call should reset the iterator and return the first batch again
    batch_1_again = solver.problem._get_next_batch()
    
    first_batch_from_new_iter = next(iter(fine_tune_problem.ft_dataloader))

    # Check that the tensors are the same
    assert torch.equal(batch_1_again[0], first_batch_from_new_iter[0].to(solver.device))
    assert torch.equal(batch_1_again[1], first_batch_from_new_iter[1].to(solver.device))


def test_stochastic_solver_forward_pass_sets_batch(fine_tune_problem):
    """
    Tests that _forward_pass calls problem.set_current_batch at each step.
    We can "spy" on the method by mocking it.
    """
    original_method = fine_tune_problem.set_current_batch
    # Spy on the set_current_batch method
    fine_tune_problem.set_current_batch = MagicMock(wraps=original_method)

    num_steps = 5
    solver = StochasticMSASolver(num_steps=num_steps, num_iterations=1, problem=fine_tune_problem)
    u_traj = torch.zeros(num_steps - 1, 1, fine_tune_problem.state_dim)

    solver._forward_pass(u_traj)
    
    # It should be called K-1 times in the forward pass loop
    assert fine_tune_problem.set_current_batch.call_count == num_steps - 1


def test_stochastic_solver_backward_pass_sets_batch(fine_tune_problem):
    """
    Tests that _backward_pass calls problem.set_current_batch at each step.
    """
    original_method = fine_tune_problem.set_current_batch
    # Spy on the set_current_batch method
    fine_tune_problem.set_current_batch = MagicMock(wraps=original_method)
    
    num_steps = 5
    solver = StochasticMSASolver(num_steps=num_steps, num_iterations=1, problem=fine_tune_problem)
    x_traj = torch.zeros(num_steps, 1, fine_tune_problem.state_dim)
    u_traj = torch.zeros(num_steps - 1, 1, fine_tune_problem.state_dim)

    solver._backward_pass(x_traj, u_traj)
    
    # It's called once for the terminal condition, then K-1 times in the loop.
    assert fine_tune_problem.set_current_batch.call_count == num_steps