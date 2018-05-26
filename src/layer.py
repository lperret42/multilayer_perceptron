import numpy as np
from src.activations import Relu, Logistic, Tanh
from math import exp

class Layer(object):
    def __init__(self, nb_neural, nb_neural_prev, activation="relu", output_layer=False):
        self.__init_activation__(activation)
        self.nb_neural = nb_neural
        self.nb_neural_prev = nb_neural_prev
        self.output_layer = output_layer

        self.weights = None

    def __init_activation__(self, activation_str):
        if activation_str == "relu":
            self.activation = Relu()
        elif activation_str == "logistic":
            self.activation = Logistic()
        elif activation_str == "tanh":
            self.activation = Tanh()

    def init(self):
        self.weights = np.random.rand(self.nb_neural, self.nb_neural_prev + 1) - 0.5

    def eval(self, X):
        X = np.insert(X, 0, -1, axis=0)
        print("X.shape:", X.shape)
        print("weights.shape:", self.weights.shape)
        print("self.output_layer:", self.output_layer)
        neurals_activation = np.array([self.activation.get_function(x) if not self.output_layer
            else exp(x) for x in self.weights.dot(X)])
        return (neurals_activation / (1 if not self.output_layer else sum(neurals_activation))).T

    def update_weights(self, errors):
        return errors
