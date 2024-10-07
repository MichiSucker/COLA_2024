from classes.LossFunction.class_LossFunction import LossFunction


class ParametricLossFunction(LossFunction):

    def __init__(self, func, p):
        self.parameter = p
        super().__init__(func=lambda x: func(x, self.parameter))

    def __add__(self, other):
        return LossFunction(func=lambda x: self.f(x) + other.f(x))

    def get_parameter(self):
        return self.parameter
