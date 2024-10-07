import torch
import torch.nn as nn

from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class HBF(nn.Module):
    """Class that implements the update-step of the heavy-ball with friction algorithm.

    Attributes
    ----------
        alpha : torch.Tensor
            step-size parameter
        beta : torch.Tensor
            momentum parameter

    Methods
    -------
        forward
            the update-step of the algorithm
        update_state
            method to update the (internal) state of the algorithm correctly
    """

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor) -> None:
        """Instantiate an HBF-object.

        :param alpha: step-size parameter
        :param beta: momentum parameter
        """
        super(HBF, self).__init__()
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:
        """The actual update-step of the algorithm.

        :param opt_algo: the optimization algorithm itself as OptimizationAlgorithm-object.
        :return: the new iterate
        """
        return (opt_algo.current_state[1] - self.alpha * opt_algo.loss_function.grad(opt_algo.current_state[1])
                + self.beta * (opt_algo.current_state[1] - opt_algo.current_state[0]))

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        """Update the internal state of the algorithm.

        :param opt_algo: the optimization algorithm itself as OptimizationAlgorithm-object.
        """
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
