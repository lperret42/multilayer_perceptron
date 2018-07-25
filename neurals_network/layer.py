import numpy as np
from neurals_network.activations import ACTIVATIONS
from toolbox.utils import get_uniform_matrix, get_normal_matrix

class Layer(object):
    def __init__(self, size, input_size, activation="relu", dtype=np.float32,
                 weights_init=("uniform", -1e-2, 1e-2),
                 biases_init=("uniform", -1e-1, 1e-1),
                 is_network_input=False,  is_network_output=False):

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
        self.__init_weights_biases_deltas(weights_init, biases_init)
        self.__set_dtype(dtype)

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

    def clean(self):
        del self.neurals
        del self.local_grad
        if not self.is_network_input:
            del self.d_weights
            del self.d_biases
            del self.d_weights_tmp
            del self.d_biases_tmp

    def __init_activation(self, activation_name):
        if activation_name not in ACTIVATIONS:
            raise Exception("{} activation is not supported".format(activation_name))
        else:
            self.activation = ACTIVATIONS[activation_name]

    def __init_weights_biases_deltas(self, weights_init, biases_init):
        if not self.is_network_input:
            w_init_str, w_param1, w_param2 = weights_init
            b_init_str, b_param1, b_param2 = biases_init
            if w_init_str == "normal":
                self.weights = get_normal_matrix((self.size, self.input_size),
                                                 w_param1, w_param2)
            elif w_init_str == "uniform":
                self.weights = get_uniform_matrix((self.size, self.input_size),
                                                 w_param1, w_param2)
            else:
                raise Exception("{} weights init is not supported".format(w_init_str))
            if b_init_str == "normal":
                self.biases = get_normal_matrix(self.size, b_param1, b_param2)
            elif b_init_str == "uniform":
                self.biases = get_uniform_matrix(self.size, b_param1, b_param2)
            else:
                raise Exception("{} biases init is not supported".format(w_init_str))
            self.d_weights = np.zeros((self.size, self.input_size))
            self.d_biases = np.zeros((self.size, 1))
            self.d_weights_tmp = []
            self.d_biases_tmp = []

    def __set_dtype(self, dtype):
        self.weights.astype(dtype, copy=False)
        self.biases.astype(dtype, copy=False)
        self.local_grad.astype(dtype, copy=False)

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

