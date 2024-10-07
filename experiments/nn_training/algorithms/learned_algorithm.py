import torch
import torch.nn as nn
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class LearnedAlgorithm(nn.Module):
    """Class that implements the update-step of the learned algorithm for the experiment of training a neural network on
    different data sets.

    Attributes
    ----------
        dim : int
            dimension of the optimization variable

    Methods
    -------
        forward
            the update-step of the algorithm
        update_state
            method to update the (internal) state of the algorithm correctly
    """

    def __init__(self, dim: int):
        """Instantiates an LearnedAlgorithm-object.

        :param dim: dimension of the optimization variable
        """
        super(LearnedAlgorithm, self).__init__()

        # Learn diagonal preconditioner for gradient and momentum term
        self.extrapolation = nn.Parameter(0.001 * torch.ones(dim))
        self.gradient = nn.Parameter(0.001 * torch.ones(dim))

        # Layer to compute the update-direction
        in_size = 4
        h_size = 20
        out_size = 1
        self.update_layer = nn.Sequential(
            nn.Conv2d(in_size, h_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(h_size, out_size, kernel_size=1),
        )

        # Layer to compute the weights for the inputs into the update_layer
        in_size = 6
        h_size = 10
        out_size = 4
        self.coefficients = nn.Sequential(
            nn.Linear(in_size, 3 * h_size),
            nn.ReLU(),
            nn.Linear(3 * h_size, 2 * h_size),
            nn.ReLU(),
            nn.Linear(2 * h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, out_size),
        )

        # For stability
        self.eps = torch.tensor(1e-10).float()

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:
        """The actual update-step of the algorithm.

        :param opt_algo: the optimization algorithm itself as OptimizationAlgorithm-object.
        :return: the new iterate
        """

        # Compute and normalize gradient
        grad = opt_algo.loss_function.grad(opt_algo.current_state[1])
        grad_norm = torch.linalg.norm(grad).reshape((1,))
        if grad_norm > self.eps:
            grad = grad / grad_norm

        # Compute and normalized momentum term
        diff = opt_algo.current_state[1] - opt_algo.current_state[0]
        diff_norm = torch.linalg.norm(diff).reshape((1,))
        if diff_norm > self.eps:
            diff = diff / diff_norm

        # Compute current and old loss (logarithmically scaled)
        loss = torch.log(opt_algo.loss_function(opt_algo.current_state[1]).detach().reshape((1,)))
        old_loss = torch.log(opt_algo.loss_function(opt_algo.current_state[0]).detach().reshape((1,)))

        # Compute weights for the inputs into the update_layer
        coefficients = self.coefficients(
            torch.concat(
                (torch.log(1 + grad_norm.reshape((1,))),
                 torch.log(1 + diff_norm.reshape((1,))),
                 torch.dot(grad, diff).reshape((1,)),
                 torch.max(torch.abs(grad)).reshape((1,)),
                 torch.tensor(opt_algo.iteration_counter).reshape((1,)),
                 loss - old_loss,)))

        # Compute update direction
        direction = self.update_layer(torch.concat((
            coefficients[0] * self.gradient * grad.reshape((1, 1, 1, -1)),
            coefficients[1] * self.extrapolation * diff.reshape((1, 1, 1, -1)),
            coefficients[2] * grad.reshape((1, 1, 1, -1)),
            coefficients[3] * diff.reshape((1, 1, 1, -1)),
            ), dim=1)).flatten()

        # Compute new iterate
        return opt_algo.current_state[-1] + direction / (torch.tensor(1 + opt_algo.iteration_counter) ** (1/2))

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        """Update the internal state of the algorithm.

        :param opt_algo: the optimization algorithm itself as OptimizationAlgorithm-object.
        """
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
