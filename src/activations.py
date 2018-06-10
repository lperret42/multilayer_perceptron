import numpy as np
from src.math import relu, d_relu, logistic, d_logistic, tanh, d_tanh,\
                     softmax, d_softmax

class Activation(object):
    def __init__(self, function, derivative, name=""):
        self.name = name
        self.function = function
        self.derivative = derivative

    def get_function(self, x):
        return self.function(x) if self.name == "softmax" else\
                        np.array([self.function(e) for e in x])

    def get_derivative(self, x):
        #print("derivative")
        #print("self.name:", self.name)
        return self.derivative(x) if self.name == "softmax" else\
                        np.array([self.derivative(e) for e in x])

class Relu(Activation):
    def __init__(self):
        self.name = "relu"
        self.function = relu
        self.derivative = d_relu

class Logistic(Activation):
    def __init__(self):
        self.name = "logistic"
        self.function = logistic
        self.derivative = d_logistic

class Tanh(Activation):
    def __init__(self):
        self.name = "tanh"
        self.function = tanh
        self.derivative = d_tanh

class Softmax(Activation):
    def __init__(self):
        self.name = "softmax"
        self.function = softmax
        self.derivative = d_softmax
