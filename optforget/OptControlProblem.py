import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Dict, Any, Type
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


class OptimControlProblem:
    """
    Abstractly encapsulates an optimal control problem.
    This class defines the structure of a continuous-time optimal control problem.

    The system dynamics are given by:
        dx/dt = f(t, x(t), u(t))
    
    The cost functional to be minimized is:
        J[u] = phi(tf, x(tf)) + integral_{t0}^{tf} L(t, x(t), u(t)) dt
    """

    def __init__(self, f: Callable[[float, torch.Tensor, torch.Tensor],
                                   torch.Tensor],
                 running_cost: Callable[[float, torch.Tensor, torch.Tensor],
                                        torch.Tensor],
                 terminal_cost: Callable[[float, torch.Tensor], torch.Tensor],
                 x0: torch.Tensor, t0: float, tf: float, control_dim: int):
        """
        Initializes the Optimal Control Problem.

        Args:
            f (Callable): The dynamics function of the system.
                          Signature: f(t, x, u) -> dx_dt
                          x shape: (N, state_dim), u shape: (N, control_dim)
                          Returns tensor of shape (N, state_dim).
            running_cost (Callable): The running cost (Lagrangian) L.
                                     Signature: running_cost(t, x, u) -> scalar_cost
            terminal_cost (Callable): The terminal cost phi.
                                      Signature: terminal_cost(t, x) -> scalar_cost
            x0 (torch.Tensor): The initial state. Shape: (N, state_dim).
            t0 (float): The initial time.
            tf (float): The final time.
            control_dim (int): The dimension of the control input.
        """
        self.f = f
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.x0 = x0
        self.t0 = t0
        self.tf = tf

        # Infer dimensions from the initial state
        if self.x0.dim() == 1:
            # If x0 is a single vector, unsqueeze to add a batch dimension
            self.x0 = self.x0.unsqueeze(0)

        self.batch_size, self.state_dim = self.x0.shape
        self.control_dim = control_dim

    def evaluate_cost(self, t_arr: torch.Tensor, x_traj: torch.Tensor,
                      u_traj: torch.Tensor) -> float:
        """
        Given a state and control trajectory, compute the total cost J.
        This uses the trapezoidal rule for numerical integration.

        Args:
            t_arr (torch.Tensor): Array of time points. Shape: (K,).
            x_traj (torch.Tensor): State trajectory. Shape: (K, N, state_dim).
            u_traj (torch.Tensor): Control trajectory. Shape: (K-1, N, control_dim).

        Returns:
            float: The total computed cost.
        """
        # Calculate running cost at each time step
        # Note: u_traj has K-1 points, corresponding to intervals [t_k, t_{k+1}]
        running_costs = torch.zeros(len(u_traj),
                                    self.batch_size,
                                    device=t_arr.device)
        for i in range(len(u_traj)):
            # Average state over the interval for better accuracy if needed,
            # but using x[i] is common for Euler-like methods.
            running_costs[i] = self.running_cost(t_arr[i].item(), x_traj[i],
                                                 u_traj[i])

        # Integrate running cost using trapezoidal rule
        dt_arr = torch.diff(t_arr)
        integral_cost = torch.sum(running_costs.squeeze() * dt_arr)

        # Calculate terminal cost
        term_cost = self.terminal_cost(self.tf, x_traj[-1])

        total_cost = integral_cost + term_cost
        return total_cost.item()


class PMPProblem(OptimControlProblem):
    """
    Extends OptimControlProblem to include necessary interfaces for PMP-based solvers.
    PMP introduces the costate (or adjoint state) 'p' and the Hamiltonian.
    """

    def __init__(self, f, running_cost, terminal_cost, x0, t0, tf,
                 control_dim):
        super().__init__(f, running_cost, terminal_cost, x0, t0, tf,
                         control_dim)

    def hamiltonian(self, t: float, x: torch.Tensor, p: torch.Tensor,
                    u: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hamiltonian H(t, x, p, u).
        H = L(t, x, u) + p^T * f(t, x, u)

        Args:
            t (float): Time.
            x (torch.Tensor): State vector. Shape: (N, state_dim).
            p (torch.Tensor): Costate vector. Shape: (N, state_dim).
            u (torch.Tensor): Control vector. Shape: (N, control_dim).

        Returns:
            torch.Tensor: The scalar value of the Hamiltonian. Shape (N,).
        """
        L = self.running_cost(t, x, u)
        f_val = self.f(t, x, u)
        # p^T * f is a dot product for each item in the batch
        pT_f = torch.sum(p * f_val, dim=1)
        return L + pT_f

    def compute_terminal_costate(self, t_f: float,
                                 x_f: torch.Tensor) -> torch.Tensor:
        """
        Computes the terminal condition for the costate vector.
        p(t_f) = d(phi)/dx |_{t_f, x_f}

        This method must be implemented by a subclass, as the derivative depends
        on the specific form of the terminal cost function.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_terminal_costate")

    def compute_costate_dynamics(self, t: float, x: torch.Tensor,
                                 p: torch.Tensor,
                                 u: torch.Tensor) -> torch.Tensor:
        """
        Computes the dynamics of the costate vector.
        dp/dt = -dH/dx

        This method must be implemented by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_costate_dynamics")

    def compute_optimal_control(self, t: float, x: torch.Tensor,
                                p: torch.Tensor) -> torch.Tensor:
        """
        Computes the optimal control u*(t) that minimizes the Hamiltonian.
        u*(t, x, p) = argmin_u H(t, x, p, u)

        This method must be implemented by a subclass, as the optimization
        depends on the specific form of the Hamiltonian.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_optimal_control")


class FineTuneForgetProblem(PMPProblem):
    """
    A specific PMP problem for mitigating catastrophic forgetting during fine-tuning.
    The state 'x' represents the model parameters.
    The control 'u' represents the update to the parameters.
    """

    def __init__(self, lambda_reg: float, c_costs: tuple, eta: float,
                 ft_dataloader: DataLoader, model: nn.Module,
                 loss_function: Callable, x_anchor: torch.Tensor, t0: float,
                 tf: float):
        """
        Initializes the Fine-Tuning Forgetting Problem.

        Args:
            lambda_reg (float): Coefficient for the running cost regularizer.
            c_costs (tuple): Coefficients (c1, c2) for the terminal cost.
            eta (float): The learning rate for the gradient descent part of the dynamics.
            x0 (torch.Tensor): The initial model parameters (theta_0).
            ft_dataloader (DataLoader): DataLoader for the fine-tuning dataset (D2).
            model (nn.Module): The model to be fine-tuned.
            loss_function (Callable): The loss function for fine-tuning.
            t0 (float): Start time.
            tf (float): End time.
        """
        self.lambda_reg = lambda_reg
        self.c1, self.c2 = c_costs
        self.eta = eta
        self.ft_dataloader = ft_dataloader
        self.model = model
        self.loss_fn = loss_function
        self.x_anchor = x_anchor.detach().clone()
        self.device = self.x_anchor.device
        state_dim = self.x_anchor.numel()
        control_dim = state_dim
        self._current_batch = None
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.param_shapes = [p.shape for _, p in self.model.named_parameters()]
        self.param_numels = [
            p.numel() for _, p in self.model.named_parameters()
        ]

        # Initialize the parent PMPProblem
        super().__init__(self._f_dynamics, self._running_cost,
                         self._terminal_cost, x_anchor, t0, tf, control_dim)

    def set_current_batch(self, batch: tuple):
        self._current_batch = batch

    def _compute_loss_l2(self, x: torch.Tensor) -> torch.Tensor:
        if self._current_batch is None:
            raise ValueError(
                "Current batch has not been set. Call set_current_batch() first."
            )
        params_list = x.squeeze(0).split(self.param_numels)
        param_dict = {
            name: p.view(shape)
            for name, p, shape in zip(self.param_names, params_list,
                                      self.param_shapes)
        }
        inputs, labels = self._current_batch
        predictions = torch.func.functional_call(self.model, param_dict,
                                                 (inputs, ))
        loss = self.loss_fn(predictions, labels)
        return loss

    def _f_dynamics(self, t: float, x: torch.Tensor,
                    u: torch.Tensor) -> torch.Tensor:
        x_requires_grad = x.detach().clone().requires_grad_(True)
        loss_l2 = self._compute_loss_l2(x_requires_grad)
        grad_l2 = torch.autograd.grad(loss_l2,
                                      x_requires_grad,
                                      create_graph=True)[0]
        return -self.eta * grad_l2 + u

    def _running_cost(self, t: float, x: torch.Tensor,
                      u: torch.Tensor) -> torch.Tensor:
        return self.lambda_reg / 2.0 * torch.sum(u * u, dim=1)

    def _terminal_cost(self, t: float, x: torch.Tensor) -> torch.Tensor:
        loss_l2 = self._compute_loss_l2(x)
        loss_l1 = 0.5 * torch.sum((x - self.x_anchor) * (x - self.x_anchor))

        return self.c1 * loss_l1 + self.c2 * loss_l2

    def compute_terminal_costate(self, t_f: float,
                                 x_f: torch.Tensor) -> torch.Tensor:
        """
        p(t_f) = d(phi)/dx |_{t_f, x_f}
        """
        x_f_with_grad = x_f.detach().clone().requires_grad_(True)
        term_cost = self.terminal_cost(t_f, x_f_with_grad)
        p_f = torch.autograd.grad(term_cost, x_f_with_grad)[0]
        return p_f

    def compute_costate_dynamics(self, t: float, x: torch.Tensor,
                                 p: torch.Tensor,
                                 u: torch.Tensor) -> torch.Tensor:
        """
        dp/dt = -dH/dx
        H = L + p^T*f = lambda/2 * ||u||^2 + p^T*u
        Since H does not depend on x, dH/dx = 0.
        Therefore, dp/dt = 0.

        If the dynamics were more complex, e.g., f(x, u), we would need to compute
        dH/dx = dL/dx + (df/dx)^T * p.
        The (df/dx)^T*p part is a Hessian-Vector Product (HVP).
        For this problem, the dynamics are simple.
        """
        x_requires_grad = x.detach().clone().requires_grad_(True)

        loss_l2 = self._compute_loss_l2(x_requires_grad)
        grad_l2 = torch.autograd.grad(loss_l2,
                                      x_requires_grad,
                                      create_graph=True)[0]

        grad_l2_dot_p = torch.dot(grad_l2.squeeze(), p.squeeze())
        hvp = torch.autograd.grad(grad_l2_dot_p,
                                  x_requires_grad,
                                  retain_graph=False)[0]

        return self.eta * hvp

    def compute_optimal_control(self, t: float, x: torch.Tensor,
                                p: torch.Tensor) -> torch.Tensor:
        """
        u* = argmin_u H(u) = argmin_u { lambda/2 * ||u||^2 + p^T*u }
        Taking derivative w.r.t u and setting to 0:
        lambda * u + p = 0  =>  u* = -p / lambda
        """
        return -p / self.lambda_reg
