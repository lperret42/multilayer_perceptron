import numpy as np
from src.activations import Relu, Logistic, Tanh, Softmax
from math import exp

DEBUG = True
DEBUG = False

class Layer(object):
    def __init__(self, size, input_size, activation="relu",
                       input_layer=False,  output_layer=False):
        self.size = size
        self.neurals = np.array([])
        self.input_size = input_size
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.weights = np.array([])
        self.biases = np.array([])
        self.weights = np.array([])
        self.local_gradients = np.array([])

        self.__init_activation(activation)
        self.__init_weights_biases_deltas()

    def __str__(self):
        ret = ""
        ret += "activation: {}\n".format(self.activation)
        ret += "size: {}\n".format(self.size)
        ret += "input_size: {}\n".format(self.input_size)
        ret += "weights.shape: {}\n".format(self.weights.shape)
        ret += "weights: {}\n".format(self.weights)
        ret += "bias: {}\n".format(self.bias)
        ret += "input_layer: {}\n".format(self.input_layer)
        ret += "output_layer: {}\n".format(self.output_layer)
        return ret

    def __init_activation(self, activation_str):
        if activation_str == "relu":
            self.activation = Relu()
        elif activation_str == "logistic":
            self.activation = Logistic()
        elif activation_str == "tanh":
            self.activation = Tanh()
        elif activation_str == "softmax":
            self.activation = Softmax()

    def __init_weights_biases_deltas(self):
        if not self.input_layer:
            c = 1
            self.weights = c * (np.random.rand(self.size, self.input_size)) - c / 2
            self.biases = (c * (np.random.rand(self.size)) - c / 2).T
            self.deltas = np.zeros((self.size, self.input_size))

    def aggregate(self, X):
        return np.array([x + self.biases for x in self.weights.dot(X).T])

    def activate(self, X):
        return np.array([self.activation.func(x) for x in X.T])

    def derivation(self, X):
        return np.array([self.activation.deriv(x) for x in X.T])

    def eval(self, X):
        if self.input_layer:
            self.neurals = np.array(X)
        else:
            self.neurals = self.activate(self.aggregate(X))
