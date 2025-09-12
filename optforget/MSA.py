from .OptControlProblem import PMPProblem, FineTuneForgetProblem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Dict, Any, Type


class MSASolver:
    """
    Implements the Method of Successive Approximations (MSA) to solve a PMPProblem.
    MSA is an iterative algorithm that successively refines a guess for the optimal control.
    """

    def __init__(self, num_steps: int, num_iterations: int,
                 problem: PMPProblem, alpha: float = 0.2):
        """
        Initializes the MSA Solver.

        Args:
            num_steps (int): Number of time discretization steps (K).
            num_iterations (int): Number of MSA iterations.
            problem (PMPProblem): The optimal control problem instance.
        """
        self.num_steps = num_steps
        self.num_iter = num_iterations
        self.problem = problem
        self.t_arr = torch.linspace(problem.t0,
                                    problem.tf,
                                    num_steps,
                                    device=problem.x0.device)
        self.dt = (problem.tf - problem.t0) / (num_steps - 1)
        self.alpha = alpha

    def _forward_pass(self, u_traj: torch.Tensor) -> torch.Tensor:
        """
        Integrates the state dynamics forward in time given a control trajectory.

        Args:
            u_traj (torch.Tensor): Control trajectory. Shape: (K-1, N, control_dim).

        Returns:
            torch.Tensor: The resulting state trajectory. Shape: (K, N, state_dim).
        """
        x_traj = torch.zeros(self.num_steps,
                             self.problem.batch_size,
                             self.problem.state_dim,
                             device=u_traj.device)
        x_traj[0] = self.problem.x0

        for k in range(self.num_steps - 1):
            t_k = self.t_arr[k]
            x_k = x_traj[k]
            u_k = u_traj[k]

            # Forward Euler method
            dx_dt = self.problem.f(t_k.item(), x_k, u_k)
            x_traj[k + 1] = x_k + self.dt * dx_dt

        return x_traj

    def _backward_pass(self, x_traj: torch.Tensor,
                       u_traj: torch.Tensor) -> torch.Tensor:
        """
        Integrates the costate dynamics backward in time.

        Args:
            x_traj (torch.Tensor): State trajectory. Shape: (K, N, state_dim).
            u_traj (torch.Tensor): Control trajectory. Shape: (K-1, N, control_dim).

        Returns:
            torch.Tensor: The resulting costate trajectory. Shape: (K, N, state_dim).
        """
        p_traj = torch.zeros_like(x_traj)

        # Set terminal condition
        p_traj[-1] = self.problem.compute_terminal_costate(
            self.problem.tf, x_traj[-1])

        # Integrate backwards from K-2 down to 0
        for k in range(self.num_steps - 2, -1, -1):
            t_k = self.t_arr[k]
            x_k = x_traj[k]
            p_k_plus_1 = p_traj[k + 1]
            u_k = u_traj[k]

            # Euler method for backward integration:
            # p[k] = p[k+1] - dt * dp/dt|_{k+1}
            # However, it's more stable to use values at step k:
            # p[k] = p[k+1] - dt * dp/dt|_k  (using p[k+1] as the reference for dp/dt)
            # This is equivalent to Forward Euler on the reversed time dynamics.
            dp_dt = self.problem.compute_costate_dynamics(
                t_k.item(), x_k, p_k_plus_1, u_k)
            p_traj[k] = p_k_plus_1 - self.dt * dp_dt

        return p_traj

    def _update_control(self, x_traj: torch.Tensor,
                        p_traj: torch.Tensor) -> torch.Tensor:
        """
        Computes the new control trajectory based on the current state and costate.

        Args:
            x_traj (torch.Tensor): State trajectory. Shape: (K, N, state_dim).
            p_traj (torch.Tensor): Costate trajectory. Shape: (K, N, state_dim).

        Returns:
            torch.Tensor: The updated control trajectory. Shape: (K-1, N, control_dim).
        """
        # u_traj has K-1 points
        control_dim = p_traj.shape[-1]  # Assuming control_dim can be inferred

        u_new_traj = torch.zeros(self.num_steps - 1,
                                 self.problem.batch_size,
                                 control_dim,
                                 device=x_traj.device)
        for k in range(self.num_steps - 1):
            u_new_traj[k] = self.problem.compute_optimal_control(
                self.t_arr[k].item(), x_traj[k], p_traj[k + 1])

        return u_new_traj

    def solve(self, u_init: torch.Tensor):
        """
        Executes the main MSA algorithm loop.

        Args:
            u_init (torch.Tensor): Initial guess for the control trajectory.
                                   Shape: (K-1, N, control_dim).

        Returns:
            tuple: A tuple containing:
                - u_optimal (torch.Tensor): The final control trajectory.
                - x_optimal (torch.Tensor): The final state trajectory.
                - costs (list): A list of total costs at each iteration.
        """
        u_traj = u_init
        costs = []

        for i in range(self.num_iter):
            # 1. Forward pass to get state trajectory
            x_traj = self._forward_pass(u_traj)

            # 2. Backward pass to get costate trajectory
            p_traj = self._backward_pass(x_traj, u_traj)

            # 3. Update the control trajectory
            u_traj_new = self._update_control(x_traj, p_traj)

            # Simple update (no line search)
            # A learning rate could be added here: u_traj = u_traj + alpha * (u_traj_new - u_traj)
            u_traj = (1 - self.alpha) * u_traj + self.alpha * u_traj_new

            # (Optional) Evaluate and store cost
            cost = self.problem.evaluate_cost(self.t_arr, x_traj, u_traj)
            costs.append(cost)
            print(f"Iteration {i+1}/{self.num_iter}, Cost: {cost:.4f}")

        # Final forward pass with the last control trajectory
        x_optimal = self._forward_pass(u_traj)

        return u_traj, x_optimal, costs

class MSAOptimizer(Optimizer):
    """
    Implements MSA as a PyTorch Optimizer.
    Each `step()` call performs one full iteration of the MSA algorithm.
    The parameters of the optimizer are the initial state x0 of the system.
    The control trajectory `u_traj` is stored in the optimizer's state.
    """

    def __init__(self, params, pmp_problem_class: Type[PMPProblem],
                 solver_class: Type[MSASolver],
                 msa_solver_params: Dict[str, Any], problem_params: Dict[str,
                                                                         Any]):
        """
        Initializes the MSA Optimizer.

        Args:
            params: Iterable of parameters to optimize (typically the initial state x0).
            pmp_problem_class (Type[PMPProblem]): The class of the PMP problem to solve (e.g., FineTuneForgetProblem).
            msa_solver_params (dict): Parameters for the MSASolver, e.g.,
                                      {'num_steps': 100, 'num_iterations': 1}.
            problem_params (dict): Parameters for the PMP problem constructor, e.g.,
                                   {'lambda_reg': 0.1, ...}.
        """
        if 'num_iterations' not in msa_solver_params:
            # Each step() is one iteration
            msa_solver_params['num_iterations'] = 1

        defaults = dict(pmp_problem_class=pmp_problem_class,
                        solver_class=solver_class,
                        msa_solver_params=msa_solver_params,
                        problem_params=problem_params)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single MSA optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # There should only be one parameter group and one parameter (the flattened model weights)
            if len(group['params']) != 1:
                raise ValueError(
                    "MSAOptimizer is designed to optimize a single parameter tensor (e.g., flattened model weights)."
                )

            x0 = group['params'][0]  # This is the initial state
            if x0.grad is not None:
                # The gradient of x0 might be used for other purposes, but not directly in MSA
                pass

            state = self.state[x0]

            # --- Initialize state (u_traj) on first step ---
            if 'u_traj' not in state:
                problem_for_init = group['pmp_problem_class'](
                    **group['problem_params'])
                control_dim = problem_for_init.control_dim
                msa_params = group['msa_solver_params']

                num_steps = msa_params['num_steps']

                batch_size = 1  # Optimizer works on a single parameter vector

                # Initial guess for control is zero
                u_init = torch.zeros(num_steps - 1,
                                     batch_size,
                                     control_dim,
                                     device=x0.device)
                state['u_traj'] = u_init

            # --- Run one MSA iteration ---
            u_current = state['u_traj']

            # Instantiate the problem and solver for the current step
            problem = group['pmp_problem_class'](**group['problem_params'])
            problem.x0 = x0.data.detach().clone().unsqueeze(0)
            solver_class = group['solver_class']
            solver = solver_class(problem=problem,
                                  **group['msa_solver_params'])

            # Run one iteration of MSA
            u_new, x_optimal, costs = solver.solve(u_current)

            # Store the updated control trajectory for the next step
            state['u_traj'] = u_new

            with torch.no_grad():
                final_state = x_optimal[-1]
                x0.copy_(final_state.squeeze(0))

        return loss
