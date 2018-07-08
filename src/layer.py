import numpy as np
from .activations import ACTIVATIONS

class Layer(object):
    def __init__(self, size, input_size,
            activation="relu", is_network_input=False,  is_network_output=False):

        self.activation_name = activation
        self.size = size
        self.neurals = np.array([])
        self.input_size = input_size
        self.is_network_input = is_network_input
        self.is_network_output = is_network_output
        self.weights = np.array([])
        self.biases = np.array([])
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

    #def __init_weights_biases_deltas(self, coef_w=0.1, coef_b=0.1):
    def __init_weights_biases_deltas(self, coef_w=0.4, coef_b=0.4):
        if not self.is_network_input:
            self.weights = (np.random.rand(self.size, self.input_size) *
                            coef_w) - coef_w / 2
            self.biases = np.matrix(
                (coef_b * (np.random.rand(self.size)) - coef_b / 2)
                ).T
            self.d_weights = np.zeros((self.size, self.input_size))
            self.d_biases = np.zeros((self.size, 1))

    def __str__(self):
        ret = ""
        ret += "activation: {}\n".format(self.activation_name)
        ret += "size: {}\n".format(self.size)
        ret += "input_size: {}\n".format(self.input_size)
        ret += "weights.shape: {}\n".format(self.weights.shape)
        ret += "weights: {}\n".format(self.weights)
        ret += "biases: {}\n".format(self.biases)
        ret += "is_network_input: {}\n".format(self.is_network_input)
        ret += "is_network_output: {}\n".format(self.is_network_output)
        return ret

