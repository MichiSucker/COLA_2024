import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable
import torch
from tqdm import tqdm
import pickle

from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion
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


def init_std_algo(x_0: torch.Tensor, stop_crit: StoppingCriterion, test_functions: list, n_train: int,
                  smoothness: torch.Tensor, strong_conv: torch.Tensor):
    """Instantiate the standard algorithm (here: HBF) with the correct parameters.

    :param x_0: starting point
    :param stop_crit: stopping criterion
    :param test_functions: test functions, that is, the test data set
    :param n_train: number of iterations during training (of learned algorithm)
    :param smoothness: smoothness constant of the loss-functions
    :param strong_conv: strong-convexity constant of the loss-functions
    :return: OptimizationAlgorithm-object with heavy-ball algorithm as implementation
    """

    # Setup worst-case optimal parameters for HBF
    alpha = 4 / ((torch.sqrt(smoothness) + torch.sqrt(strong_conv)) ** 2)
    beta = ((torch.sqrt(smoothness) - torch.sqrt(strong_conv)) /
            (torch.sqrt(smoothness) + torch.sqrt(strong_conv))) ** 2

    # Instantiate OptimizationAlgorithm-object with implementation of HBF
    std_algo = OptimizationAlgorithm(initial_state=x_0, implementation=HBF(alpha=alpha, beta=beta),
                                     stopping_criterion=stop_crit, loss_function=test_functions[0], n_max=n_train)

    return std_algo


def init_learned_algo(loading_path, x_0, stop_crit, test_functions, n_train) -> OptimizationAlgorithm:
    """Instantiate the learned algorithm and load the trained hyperparameters.

    :param loading_path: path to the trained model (with name 'model.pt'
    :param x_0: starting point
    :param stop_crit: stopping criterion
    :param test_functions: test functions, that is, the test data set
    :param n_train: number of iterations during training (of learned algorithm)
    :return: the trained optimization algorithm
    """
    learned_algo = OptimizationAlgorithm(initial_state=x_0, implementation=LearnedAlgorithm(dim=dim),
                                         stopping_criterion=stop_crit, loss_function=test_functions[0],
                                         n_max=n_train)
    learned_algo.implementation.load_state_dict(torch.load(loading_path + 'model.pt', weights_only=True))
    return learned_algo


def compute_sq_dist(iterates: NDArray, solutions: NDArray) -> NDArray:
    """Compute squared distance between the iterates and the solutions.

    :param iterates: given sequence of iterates as (n x m x d)-array
    :param solutions: points for which we want to compute the distance to as (n x d)-array
    :return: the squared distance between the iterates and the solutions as (n, m)-array
    """
    num_test_problems = iterates.shape[0]
    num_iterates = iterates.shape[1]
    return np.array([[np.linalg.norm(iterates[i, j, :] - solutions[i]) ** 2
                      for j in range(num_iterates)] for i in range(len(num_test_problems))])


def compute_data(opt_algo: OptimizationAlgorithm, std_algo: OptimizationAlgorithm, test_functions: list,
                 n_test: int, stopping_loss: float) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Compute iterates, losses, and distance to minimizer for the two algorithms.

    :param opt_algo: the learned optimization algorithm
    :param std_algo: the baseline algorithm (here: HBF)
    :param test_functions: the loss-functions for testing
    :param n_test: number of iterations on each test problem
    :param stopping_loss: loss at which the algorithm gets stopped (for numerical stability)
    :return: \1) the iterates of the learned algorithm
             2) the iterates of the baseline algorithm
             3) the losses of the learned algorithm
             4) the losses of the baseline algorithm
             5) the distance to the solution for the learned algorithm
             6) the distance to the solution for the baseline algorithm
    """

    # Compute the solution for each problem by solving the linear system.
    solutions = np.array([np.linalg.solve(f.get_parameter()['A'], f.get_parameter()['b'])
                          for f in test_functions]).reshape((len(test_functions), dim))

    # Compute the iterates and the corresponding losses for both algorithms
    iterates_pac = np.empty((len(test_functions), n_test + 1, dim))
    iterates_std = np.empty((len(test_functions), n_test + 1, dim))
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
        cur_it_pac, cur_losses_pac = compute_iterates_and_loss(algo=opt_algo, loss_func=f, stopping_loss=stopping_loss,
                                                               num_iterates=n_test, dim=dim)
        iterates_pac[i, :, :] = cur_it_pac
        losses_pac.append(cur_losses_pac)

        # Compute iterates and loss of HBF
        cur_it_std, cur_losses_std = compute_iterates_and_loss(algo=std_algo, loss_func=f, stopping_loss=stopping_loss,
                                                               num_iterates=n_test, dim=dim)
        iterates_std[i, :, :] = cur_it_std
        losses_std.append(cur_losses_std)

    # Transform the lists into numpy arrays
    losses_pac = np.array(losses_pac).reshape((len(test_functions), n_test + 1))
    losses_std = np.array(losses_std).reshape((len(test_functions), n_test + 1))

    # Compute distance to minimizer
    dist_pac = compute_sq_dist(iterates=iterates_pac, solutions=solutions)
    dist_std = compute_sq_dist(iterates=iterates_std, solutions=solutions)

    return iterates_pac, iterates_std, losses_pac, losses_std, dist_pac, dist_std


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

    # Fix starting point.
    # Note that HBF has a state-space of dimension 2*n
    x_0 = torch.zeros((2, dim))

    # Initialize HBF
    std_algo = init_std_algo(x_0=x_0, stop_crit=stop_crit, test_functions=test_functions, n_train=n_train,
                             smoothness=L_max, strong_conv=mu_min)

    # Initialize the learned algorithm and load the trained model
    opt_algo = init_learned_algo(loading_path=loading_path, x_0=x_0, stop_crit=stop_crit, test_functions=test_functions,
                                 n_train=n_train)

    # Do the actual evaluation
    iterates_pac, iterates_std, losses_pac, losses_std, dist_pac, dist_std = compute_data(
        opt_algo=opt_algo, std_algo=std_algo, test_functions=test_functions, n_test=n_test, stopping_loss=stopping_loss)

    # Estimate convergence probability on several test sets
    suff_desc_prob, emp_conv_prob = estimate_conv_prob(test_functions=test_functions,
                                                       suff_descent_property=suff_descent_property,
                                                       losses_pac=losses_pac,
                                                       n_train=n_train,
                                                       stopping_loss=stopping_loss)

    # Save data in such a way that it can be reused by the corresponding plotting function
    num_iterates = np.arange(n_test + 1)
    np.save(loading_path + 'n_train', n_train)
    np.save(loading_path + 'num_iterates', num_iterates)
    np.save(loading_path + 'losses_pac', losses_pac)
    np.save(loading_path + 'losses_std', losses_std)
    np.save(loading_path + 'iterates_pac', iterates_pac)
    np.save(loading_path + 'iterates_std', iterates_std)
    np.save(loading_path + 'dist_pac', dist_pac)
    np.save(loading_path + 'dist_std', dist_std)
    np.save(loading_path + 'suff_desc_prob', suff_desc_prob)
    np.save(loading_path + 'emp_conv_prob', emp_conv_prob)
    with open(loading_path + 'parameters_problem', 'wb') as f:
        pickle.dump(parameters_problem['test'], f)

    print("End evaluation.")
