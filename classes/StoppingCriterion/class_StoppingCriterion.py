from typing import Tuple, Callable


class StoppingCriterion:
    """Class to handle a stopping-criterion for the optimization algorithm.

    Attributes
    ----------
        func : Callable
            the stopping criterion
    """

    def __init__(self, func: Callable) -> None:
        """Instantiates a StoppingCriterion-object.

        :param func: the stopping criterion
        """
        self.criterion = func

    def __call__(self, *args: Tuple, **kwargs: dict) -> bool:
        """Evaluates the stopping criterion.

        :param args: possible non-keyword arguments of the stopping criterion
        :param kwargs: possible keyword arguments of the stopping criterion
        :return: boolean to specify whether the stopping criterion is reached
        """
        return self.criterion(*args)
