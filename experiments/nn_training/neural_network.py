import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Tuple, Callable


def polynomial_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Transform input x into higher-dimensional tensor [x**1, ..., x**degree]

    :param x: input to neural network
    :param degree: degree of polynomial features
    :return: higher-dimensional, transformed input
    """
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, degree + 1)], 1).reshape((-1, degree))


class NetStdTraining(nn.Module):
    """Class that implements the neural network which is used to perform the regression.

    Attributes
    ----------
        degree : int
            degree of polynomial features, that is, the input x gets transformed into [x**1, ..., x**degree]

    Methods
    -------
        forward
            performs the prediction with the neural network in such a way (torch.nn.functional) that we can
            back-propagate through it to train the optimization algorithm
    """

    def __init__(self, degree: int):
        """Instantiate a NetStdTraining-object, which is used to get performance of Adam.

        :param degree: degree of the polynomial features, which are used in the input of the neural network
        """
        super().__init__()
        self.degree = degree
        self.fc1 = nn.Linear(self.degree, 10 * self.degree)
        self.fc2 = nn.Linear(10 * self.degree, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the update of the neural network.

        :param x: input to neural network
        :return: prediction of neural network
        """
        x = f.relu(self.fc1(polynomial_features(x=x, degree=self.degree)))
        res = self.fc2(x)
        return res


class Net(nn.Module):
    """Class that implements the neural network which is used to perform the regression.

    Attributes
    ----------
        degree : int
            degree of polynomial features, that is, the input x gets transformed into [x**1, ..., x**degree]
        shape_param : list
            containing all the shapes (torch.Size) of the parameters of the neural network that is implemented in the
            NetStdTraining class

    Methods
    -------
        forward
            performs the prediction with the neural network in such a way (torch.nn.functional) that we can
            back-propagate through it to train the optimization algorithm
    """

    def __init__(self, degree: int, shape_parameters: list) -> None:
        """Instantiate a Net-object, which is used to train the optimization algorithm.

        :param degree: degree of the polynomial features, which are used in the input of the neural network
        :param shape_parameters: list of shapes to reproduce the same neural network as in the NetStdTraining class
        """
        super().__init__()
        self.degree = degree
        self.shape_param = shape_parameters
        self.dim_param = [torch.prod(torch.tensor(p)) for p in shape_parameters]

    def forward(self, x: torch.Tensor, neural_net_parameters: torch.Tensor) -> torch.Tensor:
        """Performs the update of the neural network in such a way that we can back-propagate through it to the
        hyperparameters of the optimization algorithm.

        :param x: input to the neural network
        :param neural_net_parameters: parameters of the neural network that are predicted by the optimization algorithm
        :returns: prediction of neural network
        """

        # Transform input into higher-dimensional object
        x = polynomial_features(x=x, degree=self.degree)

        # From the neural_net_parameters (prediction of optimization algorithm), extract the weights of the neural
        # network into the corresponding torch.nn.functional-functions. Then, perform the prediction in the usual way,
        # that is, by calling them successively.
        c = 0
        for i in range(0, len(self.dim_param), 2):

            # Extract weights and biases and reshape them correctly using self.shape_param
            weights = neural_net_parameters[c:c+self.dim_param[i]]
            weights = weights.reshape(self.shape_param[i])
            bias = neural_net_parameters[c+self.dim_param[i]:c+self.dim_param[i]+self.dim_param[i+1]]
            bias = bias.reshape(self.shape_param[i+1])

            # Compute the new update and update the counter c.
            # Here, use additional ReLU activation functions in between
            x = f.linear(input=x, weight=weights, bias=bias)
            c += self.dim_param[i] + self.dim_param[i+1]
            if len(self.shape_param) > 2 and (i+2 < len(self.dim_param)):
                x = f.relu(x)
        return x


def train_model(net: nn.Module,
                data: dict,
                criterion: Callable,
                n_it: int,
                lr: float) -> Tuple[nn.Module, list, list]:
    """Train the model (given in NetStdTraining) with Adam.

    :param net: neural network to be trained
    :param data: data set on which the network does a prediction
    :param criterion: loss-function to train the neural network (typically: MSE)
    :param n_it: number of iterations to train the network
    :param lr: learning-rate
    :return: \1) trained neural network
             2) Losses observed during training
             3) Corresponding iterates
    """

    # Initialize
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    iterates = []
    losses = []

    # Perform optimization (as usual)
    for i in range(n_it + 1):

        # Extract neural network (current iterate) into a single tensor
        iterates.append(nn_to_tensor(net))

        # Compute loss on the given data set and perform optimization step
        optimizer.zero_grad()
        loss = criterion(net(data['xes']), data['yes'])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return net, losses, iterates


def tensor_to_nn(tensor: torch.Tensor,
                 template: nn.Module) -> None:
    """Transform a given tensor (of weights) into a neural network, that is, load the weights that are predicted by the
    optimization algorithm into the neural network. This is the counterpart to nn_to_tensor().

    :param tensor: tensor that contains the weights all after each other
    :param template: template of the neural network architecture into which the weights get loaded
    """

    counter = 0
    for param in template.parameters():
        # If the parameter is updated by the (learned) optimization algorithm, then they have corresponding entries in
        # the tensor, which should be loaded into the template.
        if param.requires_grad:
            cur_size = torch.prod(torch.tensor(param.size()))
            cur_shape = param.shape
            param.data = tensor[counter:counter + cur_size].reshape(cur_shape)
            counter += cur_size


def nn_to_tensor(neural_network: nn.Module) -> torch.Tensor:
    """Take a neural network and extract its (learnable) weights into one big tensor. This is the counterpart to
    tensor_to_nn().

    :param neural_network: the neural network as nn.Module
    :return: tensor that contains all the weights
    """

    # Store all parameters as flattened tensors and then concatenate them into one big tensor
    all_params = []
    for name, param in neural_network.named_parameters():
        if param.requires_grad:
            all_params.append(param.flatten())
    x = torch.concat(all_params, dim=0).detach()

    return x
