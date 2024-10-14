from typing import Callable, Tuple
import torch


class LossFunction:
    """Class to handle general loss-functions. This also includes automatic gradient computation with backprop.

    Attributes
    ----------
        f : Callable
            the loss-function

    Methods
    -------
        grad
            Compute the gradient of func by backpropagation.
    """

    def __init__(self, func: Callable):
        """Initialization of an 'LossFunction'-object.

        :param func: the loss-function
        """
        self.f = func

    def __call__(self, x: torch.Tensor, *args: Tuple, **kwargs: dict) -> torch.Tensor:
        """Calling the loss-function, that is, evaluation it at a given point.

        :param x: the given point at which the loss-function should be evaluated
        :param args: possible non-keyword arguments of the loss-function
        :param kwargs: possible keyword arguments of the loss-function
        :return: function value at x
        """
        return self.f(x, *args, **kwargs)

    def grad(self, x: torch.Tensor, *args: Tuple, **kwargs: dict) -> torch.Tensor:
        """Compute the gradient of the loss-function at a given point (by backpropagation).

        :param x: the given point at which the gradient should be computed
        :param args: possible non-keyword arguments of the loss-function
        :param kwargs: possible keyword arguments of the loss-function
        :return: gradient at x
        """

        # Clone and detach x. Then, compute the loss at x.
        # Note that, like this, we cannot back-propagate through this gradient operation, that is, we do not use the
        # recurrent nature of the process, but treat the gradient solely as independent input.
        y = x.clone().detach().requires_grad_(True)
        f_val = self.f(y, *args, **kwargs)

        # If the gradient of y already got compute inside (!) the function call (which might be necessary sometimes)
        # do not compute it again.
        if y.grad is None:
            f_val.backward()
        return y.grad
