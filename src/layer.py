import numpy as np
from src.activations import Relu, Logistic, Tanh, Softmax
from math import exp

DEBUG = True
DEBUG = False

class Layer(object):
    def __init__(self, nb_neural, nb_neural_prev, activation="relu",
                       bias=0, input_layer=False,  output_layer=False):
                       #bias=1, input_layer=False,  output_layer=False):
        self.__init_activation__(activation)
        self.nb_neural = nb_neural
        self.neurals = np.array([])
        self.nb_neural_prev = nb_neural_prev
        self.bias = bias
        self.input_layer = input_layer
        self.output_layer = output_layer

        self.weights = np.array([])
        self.biases = np.array([])
        self.weights = np.array([])
        self.local_gradients = np.array([])

    def __str__(self):
        ret = ""
        ret += "activation: {}\n".format(self.activation)
        ret += "nb_neural: {}\n".format(self.nb_neural)
        ret += "nb_neural_prev: {}\n".format(self.nb_neural_prev)
        ret += "weights.shape: {}\n".format(self.weights.shape)
        ret += "weights: {}\n".format(self.weights)
        ret += "bias: {}\n".format(self.bias)
        ret += "input_layer: {}\n".format(self.input_layer)
        ret += "output_layer: {}\n".format(self.output_layer)
        return ret

    def __init_activation__(self, activation_str):
        if activation_str == "relu":
            self.activation = Relu()
        elif activation_str == "logistic":
            self.activation = Logistic()
        elif activation_str == "tanh":
            self.activation = Tanh()
        elif activation_str == "softmax":
            self.activation = Softmax()
        else:
            print("unknown activation function")

    def init(self):
        coef = 1
        self.weights = ((coef * (np.random.rand(self.nb_neural, self.nb_neural_prev))) - coef / 2).astype(np.float64)
        self.biases = ((coef * (np.random.rand(self.nb_neural))) - coef / 2).astype(np.float64)
        self.deltas = np.array([np.array([np.float64(0) for _ in range(self.nb_neural_prev)]) for
                _ in range(self.nb_neural)])


    def eval(self, X):
        if self.input_layer:
            self.neurals = np.array(X)
        else:
            self.neurals = np.array(self.activation.get_function(self.weights.dot(X) + self.biases))
