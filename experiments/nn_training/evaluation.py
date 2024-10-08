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


def compute_iterates(algo: OptimizationAlgorithm, num_iterates: int, dim: int) -> Tuple[NDArray, list]:
    """Compute a given number of iterates with the algorithm and the corresponding losses and gradient-norms.

    :param algo: the optimization algorithm as OptimizationAlgorithm-object.
    :param num_iterates: number of iterates
    :param dim: dimension of optimization variable
    :return: \1) the iterates
             2) the corresponding loss
    """
    iterates = np.empty((num_iterates+1, dim))
    losses = []
    for j in range(num_iterates + 1):
        iterates[j, :] = algo.current_iterate.detach().numpy()
        losses.append(algo.eval_loss().item())
        algo.step()
    return iterates, losses


def compute_sq_dist_to_point(iterates, point):
    """Compute the squared Euclidean norm between the iterates and the point.

    :param iterates: array of iterates of shape (n_iterates, dim)
    :param point: corresponding point of shape (dim, )
    :return: list of squared distances between each iterate and the point
    """
    return [torch.linalg.norm(torch.tensor(iterates[j]) - point).item() ** 2 for j in range(len(iterates))]


def approximate_stationary_point(net: nn.Module, starting_point: torch.Tensor, criterion: Callable,
                                 data: dict, num_it: int = int(1e4), lr: float = 1e-6) -> torch.Tensor:
    """Approximate stationary point by running gradient descent with a small step-size for a large number of iterations.

    :param net: network to be trained
    :param starting_point: point from which we start approximating the 'next' stationary point
    :param criterion: loss-function of the network
    :param data: dictionary containing the data set
    :param num_it: number of iterations for gradient descent
    :param lr: learning rate of gradient descent
    :return: approximation to stationary point (last iterate of gradient descent)
    """

    # Load starting point into neural net
    tensor_to_nn(tensor=starting_point, template=net)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    pbar = tqdm(range(num_it))
    pbar.set_description('Approximating stationary point')
    for _ in pbar:
        optimizer.zero_grad()
        loss = criterion(net(data['xes']), data['yes'])
        loss.backward()
        optimizer.step()
    return nn_to_tensor(net)


def setup_nn(degree: int) -> Tuple[NetStdTraining, Net, int, list]:
    """Set up the neural networks for training with Adam and for training with the learned algorithm.

    Create the same network twice, one time as a 'standard' PyTorch neural network, where parameters of the layers are
    leaf-variables in the computational graph, and one time as a standard Pytorch function (!), in which we can insert
    the prediction of the learned algorithm as parameters. Additionally, calculates the dimension of the resulting
    optimization variable, as well as the list of (the sizes of the) trainable parameters, which is needed to switch
    between the two implementations.

    :param degree: specifies the degree of polynomial features used in the neural network (see neural_network.py for
    more details)
    :return: \1) neural network as NetStdTraining-object
             2) neural network as Net-object
             3) dimension of the optimization variable (the weights of the neural network)
             4) list of the sizes of all trainable parameters
    """
    # Setup standard neural network (where parameters of the layers are leaf-variables in the computational graph).
    # This is used for
    #   - an easy understanding of the architecture
    #   - benchmark (Adam is used as standard optimization algorithm here)
    net_std = NetStdTraining(degree=degree)

    # Setup the same neural network with standard tensors (i.e. the neural network is a standard PyTorch-function),
    # such that the optimization algorithm can be trained on it.
    shape_parameters = [p.size() for p in net_std.parameters()]
    dim = sum([torch.prod(torch.tensor(s)).item() for s in shape_parameters])
    net_learned = Net(degree=degree, shape_parameters=shape_parameters)

    return net_std, net_learned, dim, shape_parameters


def compute_iter_loss_dist_learned_algo(learned_algo: OptimizationAlgorithm,
                                        num_iter: int, net_std: NetStdTraining, criterion: Callable,
                                        num_approx_stat_points: int, lr_approx_stat_points: float, dim: int
                                        ) -> Tuple[NDArray, list, list]:
    """Compute iterates, losses, and distance to 'next' stationary point for the learned algorithm.

    :param learned_algo: the learned algorithm as OptimizationAlgorithm-object
    :param num_iter: number of iterations to perform
    :param net_std: template of the neural network as NetStdTraining-object
    :param criterion: loss-function of the neural network
    :param num_approx_stat_points: number of iterations to approximate the stationary point
    :param lr_approx_stat_points: learning rate for approximating the stationary point
    :param dim: dimension of optimization variable
    :return: \1) The iterates
             2) the corresponding losses
             3) the corresponding distance to the (approx.) stationary point
    """

    # Compute iterates and corresponding losses/gradient-norms of learned algorithm.
    cur_iterates, cur_losses_pac = compute_iterates(algo=learned_algo, num_iterates=num_iter, dim=dim)

    # Approximate stationary points for learned algorithm by running gradient descent with small step-size for a
    # large number of steps. Here, make sure that the network is set correctly, that is, set it to the last
    # predicted iterate of the learned algorithm. Finally, compute the (squared) distance of the iterates to this
    # approximate stationary point.
    approx_stat_point = approximate_stationary_point(net=net_std,
                                                     starting_point=learned_algo.current_state[-1].clone(),
                                                     criterion=criterion,
                                                     data=learned_algo.loss_function.get_parameter(),
                                                     num_it=num_approx_stat_points,
                                                     lr=lr_approx_stat_points)
    cur_dist_pac = compute_sq_dist_to_point(iterates=cur_iterates, point=approx_stat_point)

    return cur_iterates, cur_losses_pac, cur_dist_pac


def compute_iter_loss_dist_std_algo(net_std: NetStdTraining, data: dict, criterion: Callable,
                                    num_iter: int, lr: float, num_approx_stat_points: int,
                                    lr_approx_stat_points: float) -> Tuple[list, list, list]:
    """Compute iterates, losses, and distance to 'next' stationary point for Adam.

    :param net_std: template of the neural network as NetStdTraining-object
    :param data: data set for training the neural network (parameter of the loss-function of the optimization algorithm)
    :param criterion: loss-function of the neural network
    :param num_iter: number of iterations to perform
    :param lr: learning rate of Adam
    :param num_approx_stat_points: number of iterations to approximate the stationary point
    :param lr_approx_stat_points: learning rate for approximating the stationary point
    :return: \1) The iterates
             2) the corresponding losses
             3) the corresponding distance to the (approx.) stationary point
    """

    # Compute iterates, losses, and gradient-norms of Adam.
    net_std, cur_losses_std, cur_iterates_std = train_model(net=net_std, data=data,
                                                            criterion=criterion, n_it=num_iter, lr=lr)

    # Again, approximate stationary points for Adam. Here, make sure that the network is set correctly, that is, as
    # the last iterate predicted by Adam. Finally, compute the (squared) distance of the iterates to this
    # approximate stationary point.
    approx_stat_point = approximate_stationary_point(net=net_std,
                                                     starting_point=cur_iterates_std[-1].clone(),
                                                     criterion=criterion,
                                                     data=data,
                                                     num_it=num_approx_stat_points,
                                                     lr=lr_approx_stat_points)
    cur_dist_std = compute_sq_dist_to_point(iterates=cur_iterates_std, point=approx_stat_point)

    return cur_iterates_std, cur_losses_std, cur_dist_std


def compute_data(test_functions: list, num_iter: int, learned_algo: OptimizationAlgorithm,
                 net_std: NetStdTraining, criterion: Callable, lr_adam: float, dim: int
                 ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Compute the iterates, the losses, the distances to the stationary points and the gradients corresponding to the
    learned algorithm and adam on the test functions.

    :param test_functions: list of test functions
    :param num_iter: number of iterations to perform
    :param learned_algo: the learned algorithm as OptimizationAlgorithm-object
    :param net_std: the neural-network template (for Adam)
    :param criterion: loss-function of the neural network
    :param lr_adam: learning rate for Adam
    :param dim: dimension of the optimization variable
    :return: \1) iterates of the learned algorithm
             2) iterates of Adam
             3) losses of the learned algorithm
             4) losses of adam
             5) distances to stationary points for the learned algorithm
             6) distances to stationary points for Adam
    """

    # Fix number of iterations and learning rate for the approximation of stationary points with gradient descent
    num_approx_stat_points, lr_approx_stat_points = int(2e3), 1e-6

    # Instantiate empty containers to store everything in
    iterates_pac = np.empty((len(test_functions), num_iter + 1, dim))
    iterates_std = np.empty((len(test_functions), num_iter + 1, dim))
    losses_pac, losses_std = [], []
    dist_pac, dist_std = [], []

    pbar = tqdm(enumerate(test_functions))
    pbar.set_description("Compute iterates")
    for i, f in pbar:

        # Reset state of the algorithm and set the new loss-function.
        learned_algo.reset_state()
        learned_algo.set_loss_function(f)

        # Compute iterates, losses, and distances for the learned algorithm
        cur_iterates, cur_losses_pac, cur_dist_pac = compute_iter_loss_dist_learned_algo(
            learned_algo=learned_algo, num_iter=num_iter, net_std=net_std, criterion=criterion,
            num_approx_stat_points=num_approx_stat_points, lr_approx_stat_points=lr_approx_stat_points, dim=dim)

        # Append results
        iterates_pac[i, :, :] = cur_iterates
        losses_pac.append(cur_losses_pac)
        dist_pac.append(cur_dist_pac)

        # Basically, perform the same for standard algorithm, that is, Adam.
        # For this, reset the neural network that is trained with Adam to the initial state.
        tensor_to_nn(learned_algo.initial_state[-1].clone(), template=net_std)
        cur_iterates_std, cur_losses_std, cur_dist_std = compute_iter_loss_dist_std_algo(
            net_std=net_std, data=f.get_parameter(), criterion=criterion, num_iter=num_iter, lr=lr_adam,
            num_approx_stat_points=num_approx_stat_points, lr_approx_stat_points=lr_approx_stat_points)

        # Append losses to lists
        iterates_std[i, :, :] = torch.stack(cur_iterates_std)
        losses_std.append(cur_losses_std)
        dist_std.append(cur_dist_std)

    # Transform everything to numpy-arrays
    losses_pac = np.array(losses_pac).reshape((len(test_functions), num_iter + 1))
    losses_std = np.array(losses_std).reshape((len(test_functions), num_iter + 1))
    dist_pac = np.array(dist_pac).reshape((len(test_functions), num_iter + 1))
    dist_std = np.array(dist_std).reshape((len(test_functions), num_iter + 1))

    return iterates_pac, iterates_std, losses_pac, losses_std, dist_pac, dist_std


def load_algorithm(loading_path: str, test_functions: list, n_train: int) -> OptimizationAlgorithm:
    """Instantiate and load the learned algorithm.

    :param loading_path: path to the trained model (with name 'model.pt'
    :param test_functions: the test functions for evaluation
    :param n_train: number of iterations the algorithm was trained on
    :return: the trained optimization algorithm
    """

    # Set initial state.
    # Note that the seed of 0 is important here: The learned algorithm only got trained from that specific
    # initialization, such that it overfits to it. Starting from another point might yield to a degradation in
    # performance. Thus: Do NOT change this!
    torch.manual_seed(0)
    x_0 = torch.randn(2 * dim).reshape((2, -1))

    # Instantiate algorithm and load its weights.
    stop_crit = GradientCriterion(eps=1e-6)
    learned_algo = OptimizationAlgorithm(
        initial_state=x_0,
        implementation=LearnedAlgorithm(dim=dim),
        stopping_criterion=stop_crit,
        loss_function=test_functions[0],
        n_max=n_train
    )
    learned_algo.implementation.load_state_dict(torch.load(loading_path + 'model.pt', weights_only=True))
    return learned_algo


def evaluate_nn(loading_path: str) -> None:
    """Evaluate the trained model on a new test-set (from the same distribution).

    :param loading_path: path where the trained model (and the other data) is stored in
    """

    print("Starting evaluation.")

    # Load data
    n_train = np.load(loading_path + 'n_train.npy')
    n_test = 2 * n_train
    conv_stat_points_property, suff_descent_property = get_property_conv_to_stationary_points(n_max=n_train)

    # Specify degree of polynomial features
    degree = 5

    # Set step-size of Adam.
    # This was found originally with grid-search. Do NOT change!
    lr_adam = 0.008
    print(f"Learning rate of Adam is set to {lr_adam}.")

    # Set up the neural networks for training with Adam and the learned algorithm, and parameters needed to change
    # between them.
    net_std, net_learned, dim, shape_parameters = setup_nn(degree=degree)

    # Get test functions (from same distribution as for training) and transform them into
    # ParametricLossFunction-objects, which can be used for training the optimization algorithm.
    # Here, we set up 2500 test functions, from which we sample again later on (uniformly).
    loss_func, criterion, parameters = get_data(neural_network=net_learned,
                                                n_prior=0, n_train=0, n_test=2500, n_val=0,
                                                n_obs_problem=50, deg_poly=degree, noise_level=1)
    test_functions = [ParametricLossFunction(func=loss_func, p=p) for p in parameters['test']]

    # Instantiate algorithm and load its weights.
    learned_algo = load_algorithm(loading_path=loading_path, test_functions=test_functions, n_train=n_train)

    # Compute iterates, losses, gradients, distance to (approximate) stationary points, etc.
    iterates_pac, iterates_std, losses_pac, losses_std, dist_pac, dist_std = compute_data(
        test_functions=test_functions, num_iter=n_test, learned_algo=learned_algo, net_std=net_std,
        criterion=criterion, lr_adam=lr_adam, dim=dim)

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
    np.save(loading_path + 'dist_pac', dist_pac)
    np.save(loading_path + 'dist_std', dist_std)
    np.save(loading_path + 'suff_desc_prob', suff_desc_prob)
    with open(loading_path + 'parameters_problem', 'wb') as file:
        pickle.dump(parameters['test'], file)

    print("End evaluation.")
