import torch
from typing import Tuple, Callable
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


def get_property_conv_to_stationary_points(n_max: int) -> Tuple[Callable, Callable]:
    """Implements the sufficient-descent property.

    :param n_max: maximal number of iterations to perform
    :return: \1) function, which checks the sufficient descent property
             2) function, which checks whether the algorithm converges to a stationary point
    """

    # Note that, from the three properties, that is,
    #   1) sufficient-descent property
    #   2) relative-error property
    #   3) boundedness property
    # only the first one has to be checked, as we can only investigate a FINITE number (n_max) of iterations.
    # Then, however, the relative-error condition always holds with max{grad-norm} / min{diff-norm}.
    # Similarly, every finite sequence is bounded. Finally, for the sufficient-descent condition, we only have to check
    # whether consecutive iterates strictly decrease the loss.

    def sufficient_descent_property(all_losses: torch.Tensor) -> bool:
        """Evaluate the sufficient-descent property.

        :param all_losses: the losses that where observed along the trajectory
        :return: boolean to specify, whether the sufficient-descent property does hold
        """

        # Specify some small constant. Below this value there might be problems with numerical instabilities.
        # Thus, only consider iterates up to this point.
        eps = 1e-16
        if len(all_losses[all_losses < eps]) >= 1:
            # Get first entry, where the loss is below the threshold eps.
            idx_first = int(torch.where(all_losses < eps)[0][0])
            # Only take losses up to this point.
            all_losses = all_losses[:idx_first]

        # Compute the pair-wise differences
        difference = all_losses[1:] - all_losses[:-1]

        # The algorithm has the sufficient-descent property, if all pair-wise differences are strictly smaller than
        # zero.
        return torch.all(difference < 0).item()

    def conv_to_stationary_points_property(f: LossFunction, opt_algo: OptimizationAlgorithm) -> bool:
        """Evaluate whether the algorithm converges to a stationary point (based on the n_max iterations).

        :param f: the loss-function the algorithm is applied to
        :param opt_algo: the optimization algorithm
        :return: boolean to specify whether the algorithm converges to a stationary point of the loss-function
        """

        # Store current state, loss function, etc. so that we can reset the algorithm to this state again.
        cur_state, cur_loss = opt_algo.current_state, opt_algo.loss_function
        cur_iteration_count = opt_algo.iteration_counter
        opt_algo.reset_state()  # 'Go to' initial state

        # Set new loss function and compute corresponding losses
        opt_algo.set_loss_function(f)
        all_losses = [opt_algo.eval_loss().item()]
        for i in range(n_max):
            # Perform a step ...
            opt_algo.step()
            # ... and compute the new loss
            all_losses.append(opt_algo.eval_loss().item())

        # Reset current state, loss function, etc.
        opt_algo.set_current_state(cur_state)
        opt_algo.set_loss_function(cur_loss)
        opt_algo.set_iteration_counter(cur_iteration_count)

        # Evaluate the sufficient-descent property. Again, note that this is sufficient, because we only can consider a
        # finite number of iterates.
        return sufficient_descent_property(torch.tensor(all_losses))

    return conv_to_stationary_points_property, sufficient_descent_property
