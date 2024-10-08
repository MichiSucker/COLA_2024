import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable
import torch
from tqdm import tqdm
import pickle
from properties_trajectory import get_property_conv_to_stationary_points
from experiments.quadratics.data_generation import get_data
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.StoppingCriterion.derived_classes.subclass_LossCriterion import LossCriterion
from experiments.quadratics.algorithms.hbf import HBF
from experiments.quadratics.algorithms.learned_algorithm import LearnedAlgorithm


def compute_iterates_and_loss(algo: OptimizationAlgorithm,
                              loss_func: ParametricLossFunction,
                              stopping_loss: float,
                              num_iterates: int,
                              dim: int) -> Tuple[NDArray, list]:
    """Compute iterates and corresponding function values for the given algorithm and loss-function.

    :param algo: the optimization algorithm
    :param loss_func: the loss-function
    :param stopping_loss: the loss threshold
    :param num_iterates: number of iterates to compute
    :param dim: the dimension of the optimization variable
    :return: the iterates and the corresponding losses
    """

    # Instantiate corresponding empty containers
    iterates = np.empty((num_iterates + 1, dim))
    losses = []
    for j in range(num_iterates + 1):
        # Compute new iterate and corresponding loss
        iterates[j, :] = algo.current_iterate.detach().numpy()
        cur_loss = loss_func(torch.tensor(iterates[j, :]))
        losses.append(cur_loss)

        # If the loss did go below the stopping loss, extend the losses with this value and stop the algorithm here
        if cur_loss < stopping_loss:
            losses.extend([cur_loss] * (num_iterates - j))
            iterates[j + 1:, :] = iterates[j, :]
            break

        # Otherwise, perform the next step
        algo.step()
    return iterates, losses


def estimate_conv_prob(test_functions: list,
                       suff_descent_property: Callable,
                       losses_pac: NDArray,
                       n_train: int,
                       stopping_loss: float) -> Tuple[NDArray, NDArray]:
    """Estimate the probability to have the sufficient-descent condition, as well as the probability to have a
    converging trajectory.

    :param test_functions: all the available functions from the test set
    :param suff_descent_property: function to check for the sufficient-descent property
    :param losses_pac: the corresponding losses of the learned algorithm
    :param n_train: number of iterations during training (for which the PAC-bound actually holds)
    :param stopping_loss: minimal loss at which the algorithm gets stopped
    :return: estimates for the probability of sufficient descent and for convergence
    """

    # Estimate convergence probability on several test sets
    suff_desc_prob, emp_conv_prob = [], []
    test_size = 250
    pbar = tqdm(range(250))
    pbar.set_description("Estimate convergence probability")
    for _ in pbar:

        # Sample indices for current 'new' test set.
        cur_idx = np.random.choice(np.arange(len(test_functions)), size=test_size, replace=False)

        # Estimate probability to observe a sufficient descent.
        # Note that we have pac_bound <= P[A_suff_desc] <= P[A_conv].
        # Further, note that the pac_bound is only valid for n_train iterations, as we could only observe n_train
        # iterations during training.
        suff_desc_prob.append(np.mean([1. if suff_descent_property(torch.tensor(losses_pac[i, :n_train + 1]))
                                       else 0. for i in cur_idx]))

        # Estimate the convergence probability: Check if the last loss is smaller than eps.
        emp_conv_prob.append(np.mean([1. if losses_pac[i, -1] < stopping_loss else 0. for i in cur_idx]))

    suff_desc_prob = np.array(suff_desc_prob).flatten()
    emp_conv_prob = np.array(emp_conv_prob).flatten()
    return np.array(suff_desc_prob).flatten(), np.array(emp_conv_prob).flatten()


def evaluate_quad(loading_path: str) -> None:
    """Evaluate the trained model for the experiment on quadratic functions.

    :param loading_path: path where the trained model is stored, and where the data gets saved into
    """

    print("Starting evaluation.")

    # Specify basic tensor types
    torch.set_default_dtype(torch.double)

    # Specify parameters as they were during training
    n_train = np.load(loading_path + 'n_train.npy')
    n_test = 2 * n_train
    conv_stat_points_property, suff_descent_property = get_property_conv_to_stationary_points(n_max=n_train)
    dim = 200
    stopping_loss = 1e-16

    # Create a new set of test-functions from the same distribution
    parameters_problem, loss_func, mu_min, L_max = get_data(dim=dim, n_prior=0, n_train=0, n_val=0, n_test=2500)
    test_functions = [ParametricLossFunction(func=loss_func, p=p) for p in parameters_problem['test']]

    # Instantiate stopping criterion
    stop_crit = LossCriterion(eps=1e-20)

    # Initialize HBF
    # State space of HBF consists of current and previous iterate
    # alpha and beta are set to the worst-case optimal choice
    x_0 = torch.zeros((2, dim))
    alpha = 4 / ((torch.sqrt(L_max) + torch.sqrt(mu_min)) ** 2)
    beta = ((torch.sqrt(L_max) - torch.sqrt(mu_min)) / (torch.sqrt(L_max) + torch.sqrt(mu_min))) ** 2
    std_algo = OptimizationAlgorithm(
        initial_state=x_0,
        implementation=HBF(alpha=alpha, beta=beta),
        stopping_criterion=stop_crit,
        loss_function=test_functions[0],
        n_max=n_train
    )

    # Initialize the learned algorithm and load the trained model
    opt_algo = OptimizationAlgorithm(
        initial_state=x_0,
        implementation=LearnedAlgorithm(dim=dim),
        stopping_criterion=stop_crit,
        loss_function=test_functions[0],
        n_max=n_train
    )
    opt_algo.implementation.load_state_dict(torch.load(loading_path + 'model.pt', weights_only=True))

    # Compute the solution for each problem by solving the linear system.
    solutions = np.array([np.linalg.solve(f.get_parameter()['A'], f.get_parameter()['b'])
                          for f in test_functions]).reshape((len(test_functions), dim))

    # Compute the iterates and the corresponding losses for both algorithms
    iterates_pac = np.empty((len(test_functions), n_test+1, dim))
    iterates_std = np.empty((len(test_functions), n_test+1, dim))
    losses_pac, losses_std = [], []

    pbar = tqdm(enumerate(test_functions))
    pbar.set_description("Compute iterates")
    for i, f in pbar:
        # Reset both algorithms to their initial state and 'give' them the new loss-function.
        opt_algo.reset_state()
        std_algo.reset_state()
        opt_algo.set_loss_function(f)
        std_algo.set_loss_function(f)

        # Compute iterates and loss of learned algorithm.
        cur_it_pac, cur_losses_pac = compute_iterates_and_loss(algo=opt_algo,
                                                               loss_func=f,
                                                               stopping_loss=stopping_loss,
                                                               num_iterates=n_test,
                                                               dim=dim)
        iterates_pac[i, :, :] = cur_it_pac
        losses_pac.append(cur_losses_pac)

        # Compute iterates and loss of HBF
        cur_it_std, cur_losses_std = compute_iterates_and_loss(algo=std_algo,
                                                               loss_func=f,
                                                               stopping_loss=stopping_loss,
                                                               num_iterates=n_test,
                                                               dim=dim)
        iterates_std[i, :, :] = cur_it_std
        losses_std.append(cur_losses_std)

    # Transform the lists into numpy arrays
    losses_pac = np.array(losses_pac).reshape((len(test_functions), n_test+1))
    losses_std = np.array(losses_std).reshape((len(test_functions), n_test+1))

    # Compute distance to minimizer
    dist_pac = np.array([[np.linalg.norm(iterates_pac[i, j, :] - solutions[i]) ** 2
                          for j in range(n_test+1)] for i in range(len(test_functions))])
    dist_std = np.array([[np.linalg.norm(iterates_std[i, j, :] - solutions[i]) ** 2
                          for j in range(n_test+1)] for i in range(len(test_functions))])
    num_iterates = np.arange(n_test+1)

    # Estimate convergence probability on several test sets
    suff_desc_prob, emp_conv_prob = estimate_conv_prob(test_functions=test_functions,
                                                       suff_descent_property=suff_descent_property,
                                                       losses_pac=losses_pac,
                                                       n_train=n_train,
                                                       stopping_loss=stopping_loss)

    # Save data in such a way that it can be reused by the corresponding plotting function
    np.save(loading_path + 'n_train', n_train)
    np.save(loading_path + 'num_iterates', num_iterates)
    np.save(loading_path + 'losses_pac', losses_pac)
    np.save(loading_path + 'losses_std', losses_std)
    np.save(loading_path + 'iterates_pac', iterates_pac)
    np.save(loading_path + 'iterates_std', iterates_std)
    np.save(loading_path + 'dist_pac', dist_pac)
    np.save(loading_path + 'dist_std', dist_std)
    np.save(loading_path + 'solutions', solutions)
    np.save(loading_path + 'suff_desc_prob', suff_desc_prob)
    np.save(loading_path + 'emp_conv_prob', emp_conv_prob)
    with open(loading_path + 'parameters_problem', 'wb') as f:
        pickle.dump(parameters_problem['test'], f)

    print("End evaluation.")
