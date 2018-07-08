import operator
import warnings
import numpy as np
from .layer import Layer
from .utils import train_test_split, prediction_precision, prediction_mean_error,\
                   mean_squared_error, cross_entropy_error,\
                   multi_to_one, is_float

class Mlp(object):
    def __init__(self, dim_input, dim_output,
                       hidden_layer_sizes=None, activation="tanh"):

        if hidden_layer_sizes is None:
            hidden_layer_sizes = (int(dim_input / 2), int(dim_input / 2), )
        self.nb_hidden_layers = len(hidden_layer_sizes)
        self.nb_layers = self.nb_hidden_layers + 2
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.__init_layers(activation, hidden_layer_sizes)

    def fit(self, X, y, learning_rate=0.1, batch_size=200,
            max_epochs=500, momentum=0.9, epsilon=1e-2,
            early_stopping=False, e_s=20, val_ratio=0.1, verbose=False):
        X, y = self.__preprocess_data(X, y)
        X_train, y_train, X_val, y_val = train_test_split(
                                    X, y, train_ratio=1 - val_ratio)
        nb_samples = X_train.shape[1]
        if verbose:
            print("nb samples:", nb_samples)
            print("dim input:", self.dim_input)
            print("dim output:", self.dim_output)
        batch_size = min(batch_size, nb_samples)
        try:
            self.__training_loop(X_train, y_train, X_val, y_val, learning_rate,
                max_epochs, momentum, nb_samples, batch_size, epsilon,
                early_stopping, e_s, verbose)
        except KeyboardInterrupt:
                    warnings.warn("Training interrupted by user.")

    def predict(self, X, need_standardize=True):
        if need_standardize:
            X = self.__standardize(np.matrix(X))
        raw_predict = self.__predict(X).T
        predict = [{self.labels[k]: raw_predict[n][k] for k in
            range(raw_predict.shape[1])} for n in range(raw_predict.shape[0])]
        return predict

    def predict_probas(self, X, need_standardize=True):
        predict = self.predict(X, need_standardize=need_standardize)
        labels = [max(p.items(), key=operator.itemgetter(1))[0] for p in predict]
        return labels

    def print_layers(self):
        for k, layer in enumerate(self.layers):
            print("layer {}:\n".format(k), layer, sep='')

    def __training_loop(self, X_train, y_train, X_val, y_val, learning_rate,
                        max_epochs, momentum, nb_samples, batch_size, epsilon,
                        early_stopping, e_s, verbose, lr_decrease=1):
        last_loss, best_loss = float("inf"), float("inf")
        weights_biases = [{"w": l.weights, "b": l.biases} for l in self.layers]
        last_improvement = 0
        epoch = 0
        try:
            while epoch <= max_epochs:
                index = np.random.choice(nb_samples, batch_size, replace=False)
                sub_samples = X_train[:, index]
                observations = y_train[:, index]
                predictions = self.__predict(sub_samples)
                errors = observations - predictions
                self.__backprop(errors)
                self.__weights_update(learning_rate, batch_size, momentum)
                val_pred = self.__predict(X_val)
                loss = cross_entropy_error(val_pred, y_val)
                if verbose:
                    print("loss at epoch {}: {}".format(epoch, round(loss, 5)))
                if loss > last_loss:
                    learning_rate *= lr_decrease
                if early_stopping:
                    if loss > best_loss:
                        last_improvement += 1
                    else:
                        weights_biases = [
                            {"w": l.weights.copy(), "b": l.biases.copy()} for
                        l in self.layers]
                        last_improvement = 0
                        best_loss = loss
                    if last_improvement >= e_s:
                        if verbose:
                            print("Early stopping")
                        val_pred = self.__predict(X_val)
                        loss = cross_entropy_error(val_pred, y_val)
                        self.force_weights_biases(weights_biases)
                        val_pred = self.__predict(X_val)
                        loss = cross_entropy_error(val_pred, y_val)
                        break
                last_loss = loss
                epoch += 1
        except KeyboardInterrupt:
                    warnings.warn("Training has been interrupted")
        if verbose:
            print("loss at end: {}".format(round(loss, 5)))

    def force_weights_biases(self, weights_biases):
        for k, w_b in enumerate(weights_biases):
            self.layers[k].weights = w_b["w"]
            self.layers[k].biases = w_b["b"]

    def __backprop(self, errors):
        for n in reversed(range(1, self.nb_layers)):
            layer = self.layers[n]
            if not layer.is_network_output:
                errors = self.layers[n+1].weights.T * self.layers[n+1].local_grad
            if layer.is_network_output:
            #if True:
                layer.local_grad = errors
            else:
                if layer.activation.name == "tanh":
                    deriv = 1 - layer.neurals ** 2
                elif layer.activation.name in ("logistic", "softmax"):
                    deriv = layer.neurals * (1 - layer.neurals)
                else:
                    deriv = layer.derivation(layer.aggregate(self.layers[n-1].neurals))
                layer.local_grad = np.multiply(errors, deriv)

    def __weights_update(self, learning_rate, batch_size, momentum):
        for n in range(1, self.nb_layers):
            layer = self.layers[n]
            layer_1 = self.layers[n - 1]
            d_weights_unit = [grad.T * neural_1.T for grad, neural_1 in
                                  zip(layer.local_grad.T, layer_1.neurals.T)]
            layer.d_weights = learning_rate * sum(d_weights_unit) / batch_size +\
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

    def __preprocess_data(self, X, y):
        mu = np.array([])
        sigma = np.array([])
        coefs = []
        X = np.matrix(X)
        y = np.array(y)
        for x in X:
            mu = np.append(mu, np.mean(x))
            sigma = np.append(sigma, np.std(x))
        X_preprocessed = ((X.T - mu) / sigma).T
        labels = np.unique(y)
        y_preprocessed = np.matrix(
            [[1 if e == label else 0 for label in labels] for e in y.T]
        ).T
        self.mu = mu
        self.sigma = sigma
        self.labels = labels
        return X_preprocessed, y_preprocessed
