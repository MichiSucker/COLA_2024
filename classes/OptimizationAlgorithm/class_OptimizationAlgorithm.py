import torch
from typing import Union
from classes.LossFunction.class_LossFunction import LossFunction


class OptimizationAlgorithm:
    """Class to handle general optimization algorithms.

    Attributes
    ----------
    initial_state : torch.Tensor
        the initialization of the algorithm
    implementation : nn.Module
        the implementation of the algorithmic update as nn.Module
    stopping_criterion : Callable
        the stopping criterion of the algorithm
    loss_function : LossFunction
        the function that should be optimized by the algorithm
    n_max : int
        maximal number of iterations to perform
    constraint : Constraint
        the constraint for the algorithm

    Methods
    -------
    set_current_state
        Update the current state of the algorithm. At least, this is computing a new iterate.
    set_iteration_counter
        Set the iteration counter to a specific number.
    set_loss_function
        Set the loss-function, which the algorithm should optimize.
    reset_state
        Reset the state of the algorithm to its initial state.
    step
        Perform a single step of the optimization algorithm. This computes a new state.
    eval_loss
        Evaluate the given loss-function at the current state.
    eval_grad
        Evaluate the gradient of the given loss-function at the current state.
    eval_constraint
        Evaluate the given constraint (possibly None) at the current state.
    """

    def __init__(self, initial_state, implementation, stopping_criterion, loss_function, n_max, constraint=None):

        # Initialize
        self.current_state = initial_state.clone()    # Current State (which can be more than just an iterate)
        self.initial_state = initial_state.clone()
        self.current_iterate = self.current_state[-1]   # Assuming that the current iterate is stored in the last row
        self.stopping_criterion = stopping_criterion    # Stopping Criterion for the Algorithm
        self.n_max = n_max
        self.implementation = implementation
        self.update_iterate = implementation.forward   # Concrete Implementation of the Update Step
        self.update_state = implementation.update_state  # Concrete Implementation of the internal state update
        self.loss_function = loss_function  # Function that is being optimized
        self.constraint = constraint
        self.dim = initial_state.shape[1]  # Dimension of Optimization Space
        self.iteration_counter = 0

    def __call__(self, x, *args, **kwargs):
        self.set_current_state(x)
        self.step()
        return self.current_state

    def __iter__(self):
        return self

    def __next__(self):

        if self.stopping_criterion(self):
            raise StopIteration
        else:
            return self.step(return_val=True)

    def set_current_state(self, new_state: torch.Tensor) -> None:
        """Sets the state of the algorithm.

        :param new_state: the state the algorithm should be set to.
        """
        self.current_state = new_state.clone()
        self.current_iterate = self.current_state[-1]

    def set_iteration_counter(self, n: int) -> None:
        """Set the iteration counter of the algorithm to a specific number

        :param n: the number
        """
        self.iteration_counter = n

    def set_loss_function(self, loss_function: LossFunction) -> None:
        """Set the loss-function, which the algorithm should optimize.

        :param loss_function: the loss-function (as LossFunction-object).
        """
        self.loss_function = loss_function

    def reset_state(self) -> None:
        """Reset the state of the algorithm to its initial state."""
        self.set_current_state(self.initial_state)
        self.set_iteration_counter(0)

    def step(self, return_val: bool = False) -> Union[None, torch.Tensor]:
        """Perform one update-iteration of the algorithm.

        :param return_val: Boolean to specify whether the new state should be returned to.
        :return: the new state (if return_val=True)
        """
        self.iteration_counter += 1
        self.current_iterate = self.update_iterate(self)
        with torch.no_grad():
            self.update_state(self)
        if return_val:
            return self.current_iterate

    def eval_loss(self) -> torch.Tensor:
        """Evaluate the given loss-function at the current state.

        :return: the function value
        """
        return self.loss_function(self.current_iterate)

    def eval_grad(self) -> torch.Tensor:
        """Evaluate the gradient of the given loss-function at the current state.

        :return: the gradient
        """
        return self.loss_function.grad(self.current_iterate)

    def eval_constraint(self) -> bool:
        """Evaluate the constraint of the algorithm.

        :return: boolean specifying whether the algorithm satisfies the constraint or not.
        """
        return self.constraint(self)
