import numpy as np
from src.activations import Relu, Logistic, Tanh, Softmax
from math import exp

DEBUG = True
DEBUG = False

class Layer(object):
    def __init__(self, nb_neural, nb_neural_prev, activation="relu", output_layer=False):
        self.__init_activation__(activation)
        self.nb_neural = nb_neural
        self.neurals = np.array([])
        self.nb_neural_prev = nb_neural_prev
        self.output_layer = output_layer

        self.weights = np.array([])
        self.local_gradients = np.array([])

    def __init_activation__(self, activation_str):
        if activation_str == "relu":
            self.activation = Relu()
        elif activation_str == "logistic":
            self.activation = Logistic()
        elif activation_str == "tanh":
            self.activation = Tanh()
        elif activation_str == "softmax":
            self.activation = Softmax()

    def init(self):
        #self.weights = np.random.rand(self.nb_neural, self.nb_neural_prev + 1) - 0.5
        self.weights = 1 * (np.random.rand(self.nb_neural, self.nb_neural_prev + 1) - 0.5)
        self.deltas = np.array([np.array([np.float64(0) for _ in range(self.nb_neural_prev + 1)]) for
                _ in range(self.nb_neural)])

    def eval(self, X):
        X = np.insert(X, 0, 0.5, axis=0)
        #print(X)
        if DEBUG:
            print("***************in eval***************************")
            print("X:", X)
            print("weights:", self.weights)
        neurals = np.array(self.activation.get_function(self.weights.dot(X)))
        #neurals = np.insert(neurals, 0, 1, axis=0)
        self.neurals = neurals
