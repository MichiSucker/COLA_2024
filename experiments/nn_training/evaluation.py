import torch
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
from experiments.nn_training.algorithms.learned_algorithm import LearnedAlgorithm
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.StoppingCriterion.derived_classes.subclass_GradientCriterion import GradientCriterion
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from properties_trajectory import get_property_conv_to_stationary_points
from experiments.nn_training.neural_network import Net, NetStdTraining, train_model, tensor_to_nn, nn_to_tensor
from experiments.nn_training.data_generation import get_data
from typing import Tuple, Callable
from numpy.typing import NDArray


def estimate_suff_desc_prob(test_functions: list,
                            suff_descent_property: Callable,
                            losses_pac: NDArray,
                            n_train: int) -> NDArray:
    """Estimate probability to observe a trajectory that satisfies the sufficient-descent condition.

    :param test_functions: list of independent test-functions
    :param suff_descent_property: function to check for the sufficient descent condition
    :param losses_pac: losses of the learned algorithm on the given test-functions
    :param n_train: number of iterations that the algorithm was trained for
    :return: estimate for the probability to observe the sufficient-descent condition (along the n_train iterations)
    """
    suff_desc_prob = []
    test_size = 250
    pbar = tqdm(range(250))
    pbar.set_description("Estimate convergence probability")
    for _ in pbar:

        # Sample indices for current 'new' test set.
        cur_idx = np.random.choice(np.arange(len(test_functions)), size=test_size, replace=False)

        # Estimate probability to observe a sufficient descent.
        suff_desc_prob.append(np.mean([1. if suff_descent_property(torch.tensor(losses_pac[i, :n_train + 1]))
                                       else 0. for i in cur_idx]))

    return np.array(suff_desc_prob).flatten()


def compute_iterates(algo, num_iterates) -> Tuple[NDArray, list, list]:
    """Compute a given number of iterates with the algorithm and the corresponding losses and gradient-norms.

    :param algo: the optimization algorithm as OptimizationAlgorithm-object.
    :param num_iterates: number of iterates
    :return: \1) the iterates
             2) the corresponding loss
             3) the corresponding gradient-norms
    """
    iterates = np.empty((num_iterates+1, dim))
    losses, grad_norms = [], []
    for j in range(num_iterates + 1):
        iterates[j, :] = algo.current_iterate.detach().numpy()
        losses.append(algo.eval_loss().item())
        grad_norms.append(torch.linalg.norm(algo.eval_grad()).item())
        algo.step()
    return iterates, losses, grad_norms


def approximate_stationary_point(net: nn.Module, criterion: Callable, data: dict,
                                 num_it: int = int(1e4), lr: float = 1e-6) -> torch.Tensor:
    """Approximate stationary point by running gradient descent with a small step-size for a large number of iterations.

    :param net: network to be trained
    :param criterion: loss-function of the network
    :param data: dictionary containing the data set
    :param num_it: number of iterations for gradient descent
    :param lr: learning rate of gradient descent
    :return: approximation to stationary point (last iterate of gradient descent)
    """
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    pbar = tqdm(range(num_it))
    pbar.set_description('Approximating stationary point')
    for _ in pbar:
        optimizer.zero_grad()
        loss = criterion(net(data['xes']), data['yes'])
        loss.backward()
        optimizer.step()
    return nn_to_tensor(net)


def evaluate_nn(loading_path: str) -> None:

    ############################################################
    # Load data
    ############################################################
    n_train = np.load(loading_path + 'n_train.npy')
    n_test = 2 * n_train
    num_approx_stat_points, lr_approx_stat_points = int(5e4), 1e-6
    conv_stat_points_property, suff_descent_property = get_property_conv_to_stationary_points(n_max=n_train)

    # Specify degree of polynomial features
    degree = 5

    # Setup standard neural network (where parameters of the layers a leaf-variables in the computational graph).
    # This is used for
    #   - an easy understanding of the architecture
    #   - benchmark (Adam is used as standard optimization algorithm here)
    net = NetStdTraining(degree=degree)

    # Setup the same neural network with standard tensors (i.e. the neural network is a standard PyTorch-function),
    # such that the optimization algorithm can be trained on it.
    shape_parameters = [p.size() for p in net.parameters()]
    dim = sum([torch.prod(torch.tensor(s)).item() for s in shape_parameters])
    neural_net = Net(degree=degree, shape_parameters=shape_parameters)

    # Get test functions (from same distribution as for training) and transform them into
    # ParametricLossFunction-objects, which can be used for training the optimization algorithm.
    loss_func, criterion, parameters = get_data(neural_network=neural_net,
                                                n_prior=0, n_train=0, n_test=2500, n_val=0,
                                                n_obs_problem=50, deg_poly=degree, noise_level=1)
    loss_functions = {
        'prior': [ParametricLossFunction(func=loss_func, p=p) for p in parameters['prior']],
        'train': [ParametricLossFunction(func=loss_func, p=p) for p in parameters['train']],
        'test': [ParametricLossFunction(func=loss_func, p=p) for p in parameters['test']],
        'validation': [ParametricLossFunction(func=loss_func, p=p) for p in parameters['validation']],
    }
    test_functions = loss_functions['test']

    # Set initial state.
    # Note that the seed of 0 is important here: The learned algorithm only got trained from that specific
    # initialization, such that it overfits to it. Starting from another point might yield to a degradation in
    # performance.
    torch.manual_seed(0)
    x_0 = torch.randn(2 * dim).reshape((2, -1))
    lr_adam = 0.008     # This was found originally with grid-search.
    print(f"Learning rate of Adam is set to {lr_adam}.")

    # Instantiate algorithm and load its weights.
    stop_crit = GradientCriterion(eps=1e-6)
    opt_algo = OptimizationAlgorithm(
        initial_state=x_0,
        implementation=LearnedAlgorithm(dim=dim),
        stopping_criterion=stop_crit,
        loss_function=test_functions[0],
        n_max=n_train
    )
    opt_algo.implementation.load_state_dict(torch.load(loading_path + 'model.pt'))

    # Instantiate empty containers to store everything in
    iterates_pac = np.empty((len(test_functions), n_test+1, dim))
    iterates_std = np.empty((len(test_functions), n_test+1, dim))
    solutions = np.empty((len(test_functions), dim))
    losses_pac, losses_std = [], []
    gradients_pac, gradients_std = [], []
    dist_pac, dist_std = [], []

    pbar = tqdm(enumerate(test_functions))
    pbar.set_description("Compute iterates")
    for i, f in pbar:

        # Reset state of the algorithm and set the new loss-function.
        opt_algo.reset_state()
        opt_algo.set_loss_function(f)

        # Compute iterates and corresponding losses/gradient-norms of learned algorithm.
        cur_iterates, cur_losses_pac, cur_grad_pac = compute_iterates(algo=opt_algo, num_iterates=n_test)

        # Approximate stationary points for learned algorithm by running gradient descent with small step-size for a
        # large number of steps. Here, make sure that the network is set correctly, that is, set it to the last
        # predicted iterate of the learned algorithm. Finally, compute the (squared) distance of the iterates to this
        # approximate stationary point.
        tensor_to_nn(opt_algo.current_state[-1].clone(), template=net)
        approx_stat_point = approximate_stationary_point(net=net,
                                                         criterion=criterion,
                                                         data=f.get_parameter(),
                                                         num_it=num_approx_stat_points,
                                                         lr=lr_approx_stat_points)
        cur_dist_pac = [torch.linalg.norm(torch.tensor(iterates_pac[i, j, :]) - approx_stat_point).item() ** 2
                        for j in range(n_test+1)]

        # Append results
        losses_pac.append(cur_losses_pac)
        gradients_pac.append(cur_grad_pac)
        dist_pac.append(cur_dist_pac)

        # Basically, perform the same for standard algorithm, that is, Adam.
        # Reset the neural network that is trained with Adam to the initial state.
        tensor_to_nn(opt_algo.initial_state[-1].clone(), template=net)

        # Compute iterates, losses, and gradient-norms of Adam.
        net, cur_losses_std, cur_iterates_std = train_model(net=net, data=f.get_parameter(), criterion=criterion,
                                                            n_it=n_test, lr=lr_adam)
        cur_grad_std = [torch.linalg.norm(f.grad(x)).item() for x in cur_iterates_std]
        iterates_std[i, :, :] = torch.stack(cur_iterates_std)

        # Again, approximate stationary points for Adam. Here, make sure that the network is set correctly, that is, as
        # the last iterate predicted by Adam. Finally, compute the (squared) distance of the iterates to this
        # approximate stationary point.
        tensor_to_nn(cur_iterates_std[-1].clone(), template=net)
        approx_stat_point = approximate_stationary_point(net=net, criterion=criterion, data=f.get_parameter(),
                                                         num_it=num_approx_stat_points, lr=lr_approx_stat_points)
        cur_dist_std = [torch.linalg.norm(torch.tensor(iterates_std[i, j, :]) - approx_stat_point).item() ** 2
                        for j in range(n_test+1)]

        # Append losses to lists
        losses_std.append(cur_losses_std)
        dist_std.append(cur_dist_std)
        gradients_std.append(cur_grad_std)

    # Transform everything to numpy-arrays
    losses_pac = np.array(losses_pac).reshape((len(test_functions), n_test + 1))
    losses_std = np.array(losses_std).reshape((len(test_functions), n_test + 1))
    dist_pac = np.array(dist_pac).reshape((len(test_functions), n_test + 1))
    dist_std = np.array(dist_std).reshape((len(test_functions), n_test + 1))
    gradients_pac = np.array(gradients_pac).reshape((len(test_functions), n_test + 1))
    gradients_std = np.array(gradients_std).reshape((len(test_functions), n_test + 1))

    # Estimate convergence probability on several test sets
    suff_desc_prob = estimate_suff_desc_prob(test_functions=test_functions, suff_descent_property=suff_descent_property,
                                             losses_pac=losses_pac, n_train=n_train)

    # Save data. Also, directly store an array with the given iterations.
    num_iterates = np.arange(n_test + 1)
    np.save(loading_path + 'n_train', n_train)
    np.save(loading_path + 'num_iterates', num_iterates)
    np.save(loading_path + 'losses_pac', losses_pac)
    np.save(loading_path + 'losses_std', losses_std)
    np.save(loading_path + 'iterates_pac', iterates_pac)
    np.save(loading_path + 'iterates_std', iterates_std)
    np.save(loading_path + 'gradients_pac', gradients_pac)
    np.save(loading_path + 'gradients_std', gradients_std)
    np.save(loading_path + 'dist_pac', dist_pac)
    np.save(loading_path + 'dist_std', dist_std)
    np.save(loading_path + 'solutions', solutions)
    np.save(loading_path + 'suff_desc_prob', suff_desc_prob)
    with open(loading_path + 'parameters_problem', 'wb') as file:
        pickle.dump(parameters['test'], file)
