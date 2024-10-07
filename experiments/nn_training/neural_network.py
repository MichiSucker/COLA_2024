import torch
import torch.nn as nn
import torch.nn.functional as f


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
