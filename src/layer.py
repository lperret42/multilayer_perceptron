import numpy as np
from src.activations import ACTIVATIONS

class Layer(object):
    def __init__(self, size, input_size,
            activation="relu", is_network_input=False,  is_network_output=False):

        self.size = size
        self.neurals = np.array([])
        self.input_size = input_size
        self.is_network_input = is_network_input
        self.is_network_output = is_network_output
        self.weights = np.array([])
        self.biases = np.array([])
        self.weights = np.array([])
        self.local_grad = np.array([])

        self.__init_activation(activation)
        self.__init_weights_biases_deltas()

    def aggregate(self, X):
        return self.weights.dot(X) + self.biases

    def activate(self, X):
        return np.concatenate([self.activation.func(x.T) for x in X.T], axis=1)

    def derivation(self, X):
        return np.concatenate([self.activation.deriv(x.T) for x in X.T], axis=1)

    def eval(self, X):
        if self.is_network_input:
            self.neurals = np.array(X)
        else:
            self.neurals = self.activate(self.aggregate(X))

    def __init_activation(self, activation_name):
        if activation_name not in ACTIVATIONS:
            raise Exception("{} activation is not supported".format(activation_name))
        else:
            self.activation = ACTIVATIONS[activation_name]

    def __init_weights_biases_deltas(self, coef=1):
        if not self.is_network_input:
            self.weights = (np.random.rand(self.size, self.input_size) *
                            coef) - coef / 2
            self.biases = np.matrix(
                (coef * (np.random.rand(self.size)) - coef / 2)
                ).T
            self.d_weights = np.zeros((self.size, self.input_size))
            self.d_biases = np.zeros((self.size, 1))

    def __str__(self):
        ret = ""
        ret += "activation: {}\n".format(self.activation)
        ret += "size: {}\n".format(self.size)
        ret += "input_size: {}\n".format(self.input_size)
        ret += "weights.shape: {}\n".format(self.weights.shape)
        ret += "weights: {}\n".format(self.weights)
        ret += "bias: {}\n".format(self.bias)
        ret += "is_network_input: {}\n".format(self.is_network_input)
        ret += "is_network_output: {}\n".format(self.is_network_output)
        return ret

