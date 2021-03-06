import numpy as np

from toolbox.maths import relu_vec, d_relu_vec, logistic_vec,\
    d_logistic_vec, tanh_vec, d_tanh_vec, heaviside_vec, d_heaviside_vec,\
    gauss_vec, d_gauss_vec, softmax, d_softmax

class Activation(object):
    def __init__(self, function, derivative, name=""):
        self.name = name
        self.function = function
        self.derivative = derivative

    def func(self, x):
        return np.asarray(self.function(x))

    def deriv(self, x):
        return np.asarray(self.derivative(x))

class Relu(Activation):
    def __init__(self):
        Activation.__init__(self, relu_vec, d_relu_vec, name="relu")

class Logistic(Activation):
    def __init__(self):
        Activation.__init__(self, logistic_vec, d_logistic_vec, name="logistic")

class Tanh(Activation):
    def __init__(self):
        Activation.__init__(self, tanh_vec, d_tanh_vec, name="tanh")

class Heaviside(Activation):
    def __init__(self):
        Activation.__init__(self, heaviside_vec, d_heaviside_vec, name="heaviside")

class Gauss(Activation):
    def __init__(self):
        Activation.__init__(self, gauss_vec, d_gauss_vec, name="gauss")

class Softmax(Activation):
    def __init__(self):
        Activation.__init__(self, softmax, d_softmax, name="softmax")

ACTIVATIONS = {
    "relu": Relu(),
    "logistic": Logistic(),
    "tanh": Tanh(),
    "heaviside": Heaviside(),
    "gauss": Gauss(),
    "softmax": Softmax(),
}
