from typing import Callable, Tuple

import torch


class Constraint:
    """Class to handle constraints of the optimization algorithm.

    Attributes
    ----------
        f : Callable
            the constraint
    """

    def __init__(self, func: Callable) -> None:
        """Instantiates a Constraint-object.

        :param func: the function that implements the constraint
        """
        self.f = func

    def __call__(self, x: torch.Tensor, *args: Tuple, **kwargs: dict) -> bool:
        """Evaluation of the constraint.

        :param x: A given point, typically the current state/iterate of the optimization algorithm.
        :param args: Possible further non-keyword arguments of the constraint.
        :param kwargs: Possible further keyword arguments of the constraint.
        :return: boolean to specify whether the constraint is satisfied at x
        """
        return self.f(x, *args, **kwargs)
