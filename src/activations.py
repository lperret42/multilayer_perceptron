import numpy as np
from src.math import relu, d_relu, logistic, d_logistic, tanh, d_tanh,\
                     softmax, d_softmax

class Activation(object):
    def __init__(self, function, derivative, name=""):
        self.name = name
        self.function = function
        self.derivative = derivative

    def func(self, x):
        return np.array(self.function(x)) if self.name == "softmax" else\
                        np.array([self.function(e) for e in x])

    def deriv(self, x):
        return self.derivative(x) if self.name == "softmax" else\
                        np.array([self.derivative(e) for e in x])

class Relu(Activation):
    def __init__(self):
        Activation.__init__(self, relu, d_relu, name="relu")

class Logistic(Activation):
    def __init__(self):
        Activation.__init__(self, logistic, d_logistic, name="logistic")

class Tanh(Activation):
    def __init__(self):
        Activation.__init__(self, tanh, d_tanh, name="tanh")

class Softmax(Activation):
    def __init__(self):
        Activation.__init__(self, softmax, d_softmax, name="softmax")
