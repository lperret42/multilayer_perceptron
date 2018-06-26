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
            self.biases = np.matrix((c * (np.random.rand(self.size)) - c / 2)).T
            self.d_weights = np.zeros((self.size, self.input_size))
            self.d_biases = np.zeros((self.size, 1))

    def aggregate(self, X):
        if DEBUG:
            print("in aggregate")
            print("dot:", self.weights.dot(X))
            print("dot.shape:", self.weights.dot(X).shape)
            print("self.biases.shape:", np.asarray(self.biases).shape)
            print("dot + biases:", self.weights.dot(X) + self.biases)
            print("end aggregate")
        return self.weights.dot(X) + self.biases

    def activate(self, X):
        if DEBUG:
            print("in activate")
            print("X:", X)
            print("X.shape:", X.shape)
            print("list x.T:",[x.T for x in X.T])
            #return np.concatenate([self.activation.func(X[:, col]) for col in range(X.shape[1])], axis=1)
            #print([self.activation.func(x.T) for x in X])
            print("after activate:", np.array([self.activation.func(x.T) for x in X.T]))
        return np.concatenate([self.activation.func(x.T) for x in X.T], axis=1)

    def derivation(self, X):
        return np.concatenate([self.activation.deriv(x.T) for x in X.T], axis=1)
        #return np.concatenate([self.activation.deriv(X[:, col]) for col in range(X.shape[1])], axis=1)

    def eval(self, X):
        if DEBUG:
            print("X before eval", X)
        if self.input_layer:
            self.neurals = np.array(X)
        else:
            self.neurals = self.activate(self.aggregate(X))
