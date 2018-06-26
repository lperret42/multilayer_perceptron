import numpy as np
from math import log
from src.layer import Layer
from src.utils import get_random_index, get_randomized_data

class Mlp(object):
    def __init__(self, dim_input, dim_output, hidden_layer_sizes=(16, 8,), activation="relu"):
        self.nb_hidden_layers = len(hidden_layer_sizes)
        self.nb_layers = self.nb_hidden_layers + 2
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.__init_layers(activation, hidden_layer_sizes)

    def __init_layers(self, activation, hidden_layer_sizes):
        self.layers = [Layer(self.dim_input, 0, activation=None, input_layer=True)]
        for hidden_layer_size in hidden_layer_sizes:
            self.layers += [Layer(
                hidden_layer_size,
                self.layers[-1].size,
                activation=activation,
            )]
        self.layers += [Layer(
            self.dim_output,
            self.layers[-1].size,
            activation="softmax",
            #activation="relu",
            output_layer=True,
        )]

    def fit(self, X, Y, learning_rate=0.5, batch_size=2, epochs=10, momentum=0.9):
        X, Y = get_randomized_data(X, Y)
        i = 0
        epoch = 0
        while epoch < epochs:
            index = get_random_index(len(X), batch_size)
            samples = np.matrix([X[i] for i in index]).T
            predictions = self.predict(samples)
            observations = np.matrix([Y[i] for i in index]).T
            errors = observations - predictions
            self.get_local_gradients(errors)
            self.update_weights(learning_rate, batch_size, momentum)
            if epoch % 100 == 0:
                print("mean_squared_error at epoch {}: {}".format(epoch, self.mean_squared_error(X, Y)))
                #rint("binary_cross_entropy_error at epoch {}: {}".format(
                #    epoch,
                #    self.binary_cross_entropy_error(X, Y),
                #))
            epoch += 1
            i = (i + 1) % len(X)

    def get_local_gradients(self, errors):
        for n in reversed(range(1, self.nb_layers)):
            layer = self.layers[n]
            layer_n_1 = self.layers[n - 1]
            if n == self.nb_layers - 1:
                deriv = layer.derivation(layer.aggregate(layer_n_1.neurals))
                layer.local_gradients = np.multiply(errors, deriv)
            else:
                deriv = layer.derivation(layer.aggregate(layer_n_1.neurals))
                errors = self.layers[n+1].weights.T.dot(self.layers[n+1].local_gradients)
                layer.local_gradients = np.multiply(errors, deriv)

    def update_weights(self, learning_rate, batch_size, momentum):
        for n in range(1, self.nb_layers):
            layer = self.layers[n]
            layer_1 = self.layers[n - 1]
            sum_product = np.sum([layer.local_gradients[:, k] * c.T for k, c in enumerate(layer_1.neurals.T)])
            layer.deltas = (learning_rate / batch_size) * sum_product + momentum * layer.deltas
            layer.weights += layer.deltas
            mean_local_gradients = layer.local_gradients.mean(axis=1)
            layer.biases += learning_rate * mean_local_gradients

    def predict(self, x):
        for k, layer in enumerate(self.layers):
            layer.eval(x)
            x = layer.neurals
            print("x in layer {}:\n".format(k), x)
        return x

    def mean_squared_error(self, X, Y):
        index = get_random_index(len(X), len(X))
        samples = np.matrix([X[i] for i in index]).T
        predictions = self.predict(samples)
        observations = np.matrix([Y[i] for i in index]).T
        errors = observations - predictions
        return (1 / errors.shape[1]) * sum([e.dot(e.T) for e in errors.T])

    def binary_cross_entropy_error(self, X, Y):
        if self.dim_output != 2:
            raise Exception("Can't call binary_cross_entropy_error if dim_output != 2")
        probas = [self.predict(x) for x in X]
        errors = [y[1] * log(p[1]) + (1 - y[1]) * log(1 - p[1]) for p, y in zip(probas, Y)]
        return (-1 / len(errors)) * sum(errors)

    def predict_label(self, x):
        predict = list(self.predict(x))
        return predict.index(max(predict))

    def show_one_result(self, x, y):
        prediction = self.predict(x)
        print("prediction: {}    real: {}".format(prediction, y))

    def print_weights(self):
        for k, layer in enumerate(self.layers):
            print("weights in layer {}".format(k), layer.weights)

    def print_layers(self):
        for k, layer in enumerate(self.layers):
            print("layer {}:\n".format(k), layer)

    def get_precision(self, X, Y):
        nb_errors = 0.
        for i in range(len(X)):
            x = X[i]
            y = list(Y[i])
            predict_label = self.predict_label(x)
            real_label = y.index(max(y))
            if predict_label != real_label:
                nb_errors += 1
        return 1 - nb_errors / len(X)
