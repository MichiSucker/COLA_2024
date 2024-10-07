import torch
from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion


class GradientCriterion(StoppingCriterion):
    """Class to handle stopping-criteria based on the gradient-norm. This class inherits from class StoppingCriterion.

    Attributes
    ----------
        eps : float
            the threshold
    """

    def __init__(self, eps):
        """Instantiate a LossCriterion-object by defining a StoppingCriterion through the norm of the gradient.

        :param eps: the threshold for the norm of the gradient
        """
        self.eps = eps
        super().__init__(lambda opt_algo:
                         torch.linalg.norm(opt_algo.loss_function.grad(opt_algo.current_iterate)) < self.eps)
