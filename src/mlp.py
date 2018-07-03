import operator
import numpy as np
from math import log
from src.layer import Layer
from src.utils import train_test_split

class Mlp(object):
    def __init__(self, dim_input, dim_output,
                       hidden_layer_sizes=None, activation="tanh"):

        if hidden_layer_sizes is None:
            hidden_layer_sizes = (int(dim_input / 2),)
        self.nb_hidden_layers = len(hidden_layer_sizes)
        self.nb_layers = self.nb_hidden_layers + 2
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.__init_layers(activation, hidden_layer_sizes)

    def fit(self, X, Y, learning_rate=0.3, batch_size=32,
                        epochs=400, momentum=0.9, verbose=False):
        X, Y = self.__preprocess_data(X, Y)
        #X_train, Y_train, X_test, Y_test = train_test_split(X, Y, train_ratio=0.9)
        #nb_sample = X_train.shape[1]
        nb_sample = X.shape[1]
        print("nb_sample:", nb_sample)
        batch_size = min(batch_size, nb_sample)
        epoch = 0
        while epoch <= epochs:
            #print("epoch:", epoch)
            index = np.random.choice(nb_sample, batch_size, replace=False)
            #print("index:", index)
            #sub_samples = X_train[:, index]
            #observations = Y_train[:, index]
            sub_samples = X[:, index]
            observations = Y[:, index]
            predictions = self.__predict(sub_samples)
            errors = observations - predictions
            self.__get_local_grad(errors)
            self.__update_weights(learning_rate, batch_size, momentum)
            if verbose and epoch != 0 and epoch % 100 == 0:
                #predictions = self.__predict(X_test)
                predictions = self.__predict(X)
                #squared_error = round(self.mean_squared_error(predictions, Y_test), 3)
                #entropy_error = round(self.cross_entropy_error(predictions, Y_test), 3)
                squared_error = round(self.mean_squared_error(predictions, Y), 3)
                entropy_error = round(self.cross_entropy_error(predictions, Y), 3)
                print("mean squared error at epoch {}: {}".format(
                                                epoch, squared_error))
                print("entropy error at epoch {}: {}".format(epoch, entropy_error))
            epoch += 1
        return

    def predict(self, X):
        X = self.__standardize(np.matrix(X))
        raw_predict = self.__predict(X).T
        predict = [{self.labels[k]: raw_predict[n][k] for k in
            range(raw_predict.shape[1])} for n in range(raw_predict.shape[0])]
        return predict

    def predict_labels(self, X):
        predict = self.predict(X)
        labels = [max(p.items(), key=operator.itemgetter(1))[0] for p in predict]
        return labels

    def get_precision(self, predictions, observations):
        return sum([1 if pred == obs else 0 for pred, obs in
            zip(predictions, observations)]) / len(predictions)

    def get_mean_error(self, predictions, observations):
        return np.mean([abs(pred - obs) for pred, obs in
            zip(predictions, observations)])

    def mean_squared_error(self, predictions, observations):
        errors = observations - predictions
        return float((1 / errors.shape[1]) * sum([e.dot(e.T) for e in errors.T]))

    def cross_entropy_error(self, predictions, observations):
        log_pred = np.vectorize(log)(predictions)
        product = np.multiply(observations, log_pred)
        return float(-sum(product.mean(axis=1)))

    def print_layers(self):
        for k, layer in enumerate(self.layers):
            print("layer {}:\n".format(k), layer)

    def __get_local_grad(self, errors):
        for n in reversed(range(1, self.nb_layers)):
            layer = self.layers[n]
            if layer.activation.name == "tanh":
                deriv = 1 - layer.neurals ** 2
            elif layer.activation.name in ("logistic", "softmax"):
                deriv = layer.neurals * (1 - layer.neurals)
            else:
                deriv = layer.derivation(layer.aggregate(self.layers[n-1].neurals))
            if not layer.is_network_output:
                errors = self.layers[n+1].weights.T * self.layers[n+1].local_grad
            layer.local_grad = np.multiply(errors, deriv)

    def __update_weights(self, learning_rate, batch_size, momentum):
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

    def __standardize(self, X):
        return ((X.T - self.mu) / self.sigma).T

    def __preprocess_data(self, X, Y):
        mu = np.array([])
        sigma = np.array([])
        coefs = []
        X = np.matrix(X)
        Y = np.array(Y)
        for x in X:
            mu = np.append(mu, np.mean(x))
            sigma = np.append(sigma, np.std(x))
        X_preprocessed = ((X.T - mu) / sigma).T
        labels = np.unique(Y)
        Y_preprocessed = np.matrix(
            [[1 if y == label else 0 for label in labels] for y in Y.T]
        ).T
        self.mu = mu
        self.sigma = sigma
        self.labels = labels
        return X_preprocessed, Y_preprocessed
