from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion


class LossCriterion(StoppingCriterion):
    """Class to handle stopping-criteria based on the function-values. This class inherits from class StoppingCriterion.

    Attributes
    ----------
        eps : float
            the threshold
    """

    def __init__(self, eps: float) -> None:
        """Instantiate a LossCriterion-object by defining a StoppingCriterion through the loss-function.

        :param eps: the threshold for the loss
        """
        self.eps = eps
        super().__init__(lambda opt_algo: opt_algo.loss_function(opt_algo.current_iterate) < self.eps)
