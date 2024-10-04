import torch
import torch.nn as nn
from typing import Callable, Tuple


def get_data(neural_network: nn.Module,
             n_prior: int,
             n_train: int,
             n_test: int,
             n_val: int,
             n_obs_problem: int = 50,
             deg_poly: int = 5,
             noise_level: float = 1.) -> Tuple[Callable, Callable, dict]:
    """Generate data for the neural-network-training experiment.

    :param neural_network: the neural network to be trained
    :param n_prior: number of loss-functions for construction of the prior
    :param n_train: number of loss-functions for PAC-Bayes training
    :param n_test: number of loss-functions for evaluation
    :param n_val: number of loss-functions for probabilistic constraint
    :param n_obs_problem: number of datapoints per problem that the neural network can observe (default: 50)
    :param deg_poly: degree of polynomials being constructed (default: 5)
    :param noise_level: standard-deviation of noise added to observations (zero-mean normal distribution)

    :returns: \1. template of the loss-function of the optimization algorithm
             2. loss-function of the neural network
             3. dictionary containing the parameters of the loss-functions as lists (separated into 'prior',
    'train', etc.)
    """

    # Define loss function of neural network
    criterion = nn.MSELoss()

    # Specify distributions of the datapoints observed by the neural network (x-values) and distribution of coefficients
    # of the polynomials.
    datapoint_distribution = torch.distributions.uniform.Uniform(-2., 2.)
    coefficient_distribution = torch.distributions.uniform.Uniform(-5, 5)

    # Get all power of the polynomials (+1 because last values if exclusive).
    powers = torch.arange(deg_poly + 1)

    # Instantiate empty dictionary for storing all the functions as separate lists (of parameters).
    # Then, fill each of these lists by creating the x- and y-values as y = g(x) + eps.
    # For this:
    #   1) Samples the x-values from the datapoint_distribution.
    #   2) Create the polynomial by sampling its coefficients.
    #   3) Evaluate the polynomials at the x-values.
    #   4) Create the y-values by adding zero-mean gaussian noise with standard deviation of noise_level.
    parameters = {'prior': [], 'train': [], 'test': [], 'validation': []}
    for cur_n_functions, name in [(n_prior, 'prior'), (n_train, 'train'), (n_test, 'test'), (n_val, 'validation')]:
        for _ in range(cur_n_functions):

            # Create observations:
            # Create x-values
            xes = datapoint_distribution.sample(torch.Size(n_obs_problem, ))
            xes, _ = torch.sort(xes)    # Sort them already now. This is needed, at least, for plotting.
            xes = xes.reshape((-1, 1))

            # Create polynomials and evaluate them at x-values
            coefficients = coefficient_distribution.sample(torch.Size(deg_poly, ))
            gt = torch.sum(coefficients * (xes ** powers), dim=1).reshape((-1, 1))

            # Create y-values
            yes = gt + noise_level * torch.randn((n_obs_problem, 1))

            # Store the result in dictionary. These are the parameters of the loss-function. Hence, it has to be stored
            # in such a way that it can be accessed correctly in the loss function (specified below).
            parameters[name].append({'xes': xes, 'yes': yes,
                                     'coefficients': coefficients, 'gt': gt,
                                     'opt_val': torch.tensor(0.0)})

    # Define the loss-function of the algorithm.
    # This is the concatenation of the loss-function of the neural network (criterion) with the neural network itself.
    def loss_function(x: torch.Tensor, parameter: dict) -> torch.Tensor:
        """This is the loss-function of the optimization algorithm.
        :param x: the parameters of the neural network, that is, the optimization-variable for the algorithm
        :param parameter: the parameters of the loss-function of the optimization variable (here: the data set)
        :return: the loss of the neural network with parameters x on the given dataset stored in parameter
        """
        return criterion(neural_network(x=parameter['xes'], neural_net_parameters=x), parameter['yes'])

    return loss_function, criterion, parameters
