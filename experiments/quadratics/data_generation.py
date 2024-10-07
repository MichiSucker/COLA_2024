import torch
from typing import Tuple, Callable


def get_data(dim: int,
             n_prior: int, n_train: int, n_val: int, n_test: int) -> Tuple[dict, Callable, torch.Tensor, torch.Tensor]:
    """Generate data for the experiment on quadratic functions.

    :param dim: dimension of the optimization variable.
    :param n_prior: number of loss-functions for construction of the prior
    :param n_train: number of loss-functions for PAC-Bayes training
    :param n_test: number of loss-functions for evaluation
    :param n_val: number of loss-functions for probabilistic constraint
    :return: \1. dictionary containing the parameters of the loss-functions as lists (separated into 'prior',
    'train', etc.)
              2. the template for the loss-function of the optimization algorithm
    """

    # Create distributions of strong-convexity and smoothness constants.
    # These are just samples uniformly from some intervals.
    # This is very specific. However, the algorithm does not "see" this structure directly, that is, it can leverage it
    # to accelerate, if it is able to find it, but we do not hard-code this information.
    mu_min, mu_max = torch.tensor(1e-3), torch.tensor(5e-3)
    str_conv_dist = torch.distributions.uniform.Uniform(mu_min, mu_max)
    L_min, L_max = torch.tensor(1e2), torch.tensor(5e2)
    smooth_dist = torch.distributions.uniform.Uniform(L_min, L_max)

    # Create distribution of right-hand sides: First, sample a mean and a matrix C randomly.
    # Then, set cov = C^T @ C to make it positive semi-definite.
    mean = torch.distributions.uniform.Uniform(-5, 5).sample(torch.Size(dim, ))
    cov = torch.distributions.uniform.Uniform(-5, 5).sample(torch.Size(dim, dim))
    cov = torch.transpose(cov, 0, 1) @ cov
    rhs_dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)

    # Instantiate dictionary for parameters (of the loss-functions of the algorithm).
    parameters = {}
    for cur_n_functions, name in [(n_prior, 'prior'), (n_train, 'train'), (n_val, 'validation'), (n_test, 'test')]:

        # Sample strong convexity and smoothness constants from the corresponding distributions.
        samples_strong_convexity = str_conv_dist.sample(torch.Size(cur_n_functions, ))
        samples_smoothness = smooth_dist.sample(torch.Size(cur_n_functions, ))

        # Create diagonals of quadratic matrix.
        diagonals = [torch.linspace(torch.sqrt(strong_convexity).item(), torch.sqrt(smoothness).item(), dim)
                     for strong_convexity, smoothness in zip(samples_strong_convexity, samples_smoothness)]

        # Sample right-hand side from the corresponding distribution.
        rhs = rhs_dist.sample(torch.Size(cur_n_functions,))
        parameters[name] = [{'A': torch.diag(diagonals[i]), 'b': rhs[i, :], 'opt_val': torch.tensor(0.0)}
                            for i in range(cur_n_functions)]

    # Specify loss function of the optimization algorithm as squared norm.
    # Parameters are the operator and the right-hand side.
    def loss_function(x, parameter):
        return 0.5 * torch.linalg.norm(torch.matmul(parameter['A'], x) - parameter['b']) ** 2

    return parameters, loss_function, mu_min, L_max
