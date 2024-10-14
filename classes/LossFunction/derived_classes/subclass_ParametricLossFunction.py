from classes.LossFunction.class_LossFunction import LossFunction
from typing import Callable


class ParametricLossFunction(LossFunction):
    """Class to handle parametric loss-functions. This class inherits from class LossFunction.

    Attributes
    ----------
        parameter : dict
            parameter of the loss-function

    Methods
    -------
        get_parameter
            Access the parameter of the loss-function.
    """

    def __init__(self, func: Callable, p: dict) -> None:
        """Initialize a ParametricLossFunction-object.

        :param func: the loss-function
        :param p: the parameter
        """
        self.parameter = p
        super().__init__(func=lambda x: func(x, self.parameter))

    def get_parameter(self) -> dict:
        """Access the parameter of the loss-function."""
        return self.parameter
