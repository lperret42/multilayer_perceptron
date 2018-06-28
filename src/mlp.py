import operator
import numpy as np
from math import log
from src.layer import Layer
from src.utils import get_random_index, get_randomized_data

class Mlp(object):
    def __init__(self, dim_input, dim_output,
                       hidden_layer_sizes=(16,), activation="tanh"):

        self.nb_hidden_layers = len(hidden_layer_sizes)
        self.nb_layers = self.nb_hidden_layers + 2
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.__init_layers(activation, hidden_layer_sizes)

    def fit(self, X, Y, learning_rate=0.7, batch_size=100, epochs=5000, momentum=0.8):
        X, Y = self.__preprocess_data(X, Y)
        nb_sample = X.shape[1]
        epoch = 0
        while epoch < epochs:
            index = get_random_index(nb_sample, batch_size)
            sub_samples = X[:, index]
            observations = Y[:, index]
            predictions = self.__predict(sub_samples)
            errors = observations - predictions
            self.get_local_grad(errors)
            self.update_weights(learning_rate, batch_size, momentum)
            if epoch != 0 and epoch % 100 == 0:
                print("mean_squared_error at epoch {}: {}".format(epoch, self.mean_squared_error(X, Y)))
                #print("binary_cross_entropy_error at epoch {}: {}".format(
                #    epoch,
                #    self.binary_cross_entropy_error(np.matrix(X).T, Y),
                #))
            epoch += 1
        return
        exit()
        print(self.predict(X))
        print(self.predict_labels(X))
        for n in range(Y.shape[1]):
            predict = self.predict(X[:, n])
            print("predict n : ",
            [round(predict[0][0], 2), round(predict[1][0], 2)],
            "    real:",
            float(Y[:, n][0][0]), float(Y[:, n][1][0]),
            )


    def get_local_grad(self, errors):
        for n in reversed(range(1, self.nb_layers)):
            layer = self.layers[n]
            deriv = layer.derivation(layer.aggregate(self.layers[n-1].neurals))
            if not layer.is_network_output:
                errors = self.layers[n+1].weights.T * self.layers[n+1].local_grad
            layer.local_grad = np.multiply(errors, deriv)

    def update_weights(self, learning_rate, batch_size, momentum):
        for n in range(1, self.nb_layers):
            layer = self.layers[n]
            layer_1 = self.layers[n - 1]
            d_weights_by_batch = [grad.T * neural_1.T for grad, neural_1 in
                                  zip(layer.local_grad.T, layer_1.neurals.T)]
            layer.d_weights = learning_rate * sum(d_weights_by_batch) / batch_size +\
                             momentum * layer.d_weights
            mean_local_grad = layer.local_grad.mean(axis=1)
            layer.d_biases = learning_rate * mean_local_grad +\
                            momentum * layer.d_biases

            layer.weights += layer.d_weights
            layer.biases += layer.d_biases

    def __predict(self, X):
        for k, layer in enumerate(self.layers):
            layer.eval(X)
            X = layer.neurals
        return X

    def mean_squared_error(self, X, Y):
        predictions = self.__predict(X)
        observations = Y
        errors = observations - predictions
        return (1 / errors.shape[1]) * sum([e.dot(e.T) for e in errors.T])

    def binary_cross_entropy_error(self, X, Y):
        if self.dim_output != 2:
            raise Exception("Can't call binary_cross_entropy_error if dim_output != 2")
        probas = self.predict(X)
        errors = [y[1] * log(p[1]) + (1 - y[1]) * log(1 - p[1]) for p, y in zip(probas, Y)]
        return (-1 / len(errors)) * sum(errors)

    def predict(self, X):
        raw_predict = self.__predict(X).T
        predict = [{self.labels[k]: raw_predict[n][k] for k in
            range(raw_predict.shape[1])} for n in range(raw_predict.shape[0])]
        return predict

    def predict_labels(self, X):
        predict = self.predict(X)
        labels = [max(p.items(), key=operator.itemgetter(1))[0] for p in predict]
        return labels

    def show_one_result(self, x, y):
        prediction = self.predict(x)
        print("prediction: {}    real: {}".format(prediction, y))

    def print_layers(self):
        for k, layer in enumerate(self.layers):
            print("layer {}:\n".format(k), layer)

    def get_precision(self, predictions, observations):
        return sum([1 if pred == obs else 0 for pred, obs in
            zip(predictions, observations)]) / len(predictions)

    def __init_layers(self, activation, hidden_layer_sizes):
        self.layers = [Layer(self.dim_input, 0, is_network_input=True)]
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
            is_network_output=True,
        )]

    def standardize(self, X):
        return ((X.T - self.mu) / self.sigma).T

    def __preprocess_data(self, X, Y):
        mu = np.array([])
        sigma = np.array([])
        coefs = []
        X = np.matrix(X)
        for x in X:
            mu = np.append(mu, np.mean(x))
            sigma = np.append(sigma, np.std(x))
        X_preprocessed = ((X.T - mu) / sigma).T
        labels = list(set(Y))
        print("labels:", labels)
        Y_preprocessed = np.matrix(
            [[1 if y == label else 0 for label in labels] for y in Y]
        ).T
        self.mu = mu
        self.sigma = sigma
        self.labels = labels
        return X_preprocessed, Y_preprocessed
