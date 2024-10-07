import torch
import torch.nn as nn

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class LearnedAlgorithm(nn.Module):
    """Class that implements the update-step of the learned algorithm for the experiment on quadratic functions.

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

    def __init__(self, dim: int) -> None:
        """Instantiates an LearnedAlgorithm-object.

        :param dim: dimension of the optimization variable
        """
        super(LearnedAlgorithm, self).__init__()

        self.dim = dim

        # Specify the update-layer
        in_size = 3
        h_size = 10
        out_size = 1
        self.update_layer = nn.Sequential(
            nn.Conv2d(in_size, 3 * h_size, kernel_size=1, bias=False),
            nn.Conv2d(3 * h_size, 3 * h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(3 * h_size, 2 * h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2 * h_size, 1 * h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.Conv2d(h_size, out_size, kernel_size=1, bias=False),
        )

        # Specify the layer which computes the step-size
        in_size = 4
        h_size = 10
        out_size = 1
        self.step_size_layer = nn.Sequential(
            nn.Linear(in_size, 3 * h_size, bias=False),
            nn.Linear(3 * h_size, 3 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(3 * h_size, 2 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(2 * h_size, 1 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.Linear(h_size, out_size, bias=False),
        )

        # For stability
        self.eps = torch.tensor(1e-20).float()

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:
        """The actual update-step of the algorithm.

        :param opt_algo: the optimization algorithm itself as OptimizationAlgorithm-object.
        :return: the new iterate
        """

        # Compute gradient and normalize it
        grad = opt_algo.loss_function.grad(opt_algo.current_state[1])
        grad_norm = torch.linalg.norm(grad).reshape((1,))
        if grad_norm > self.eps:
            grad = grad / grad_norm

        # Compute momentum term and normalize it
        diff = opt_algo.current_state[1] - opt_algo.current_state[0]
        diff_norm = torch.linalg.norm(diff).reshape((1,))
        if diff_norm > self.eps:
            diff = diff / diff_norm

        # Get update direction with update_layer
        direction = self.update_layer(torch.concat((
            grad.reshape((1, 1, 1, -1)),
            diff.reshape((1, 1, 1, -1)),
            (grad * diff).reshape((1, 1, 1, -1)),
        ), dim=1)).flatten()

        # Get step-size with step_size_layer
        step_size = self.step_size_layer(
            torch.concat(
                (torch.log(1 + grad_norm.reshape((1,))),
                 torch.log(1 + diff_norm.reshape((1,))),
                 torch.log(1 + opt_algo.loss_function(opt_algo.current_state[1]).detach().reshape((1,))),
                 torch.log(1 + opt_algo.loss_function(opt_algo.current_state[0]).detach().reshape((1,))),
                 ))
        )

        # Perform the update
        return opt_algo.current_state[-1] - step_size * direction

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        """Update the internal state of the algorithm.

        :param opt_algo: the optimization algorithm itself as OptimizationAlgorithm-object.
        """
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
