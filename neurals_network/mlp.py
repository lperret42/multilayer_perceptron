try:
    import _pickle as pickle
except:
    import pickle
import sys, os, operator, warnings, datetime
from pkg_resources import resource_filename
import numpy as np

from neurals_network.layer import Layer
from toolbox.utils import train_test_split, split_batch, cross_entropy_loss

PYTHON_MAJOR = sys.version_info[0]

class MlpClassifier(object):
    """Multilayer perceptron for multi classification.
    The training is done using gradient descent with batch to minimize the
    cross entropy loss.
    """
    def __init__(self, hidden_layer_sizes=None):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, y, activation="tanh", learning_rate=0.1, batch_size=1024,
            max_epochs=256, momentum=0.9, early_stopping=32, val_ratio=0.1,
            standardize=True, divider=1, rm_ct_features=True,
            dtype=np.float64, splitted_batch_size=2048, verbose=False):
        self.rm_ct_features = rm_ct_features
        self.dim_input = self.__get_dim_input(X, rm_ct_features=rm_ct_features)
        self.dim_output = self.__get_dim_output(y)
        self.dtype = dtype
        self.__init_layers(activation)
        X, y = self.__preprocess_train(X, y,
                                      standardize=standardize, divider=divider)
        X_train, y_train, X_val, y_val = train_test_split(
                                    X, y, train_ratio= 1 - val_ratio)
        nb_samples = X_train.shape[1]
        if verbose:
            print("nb samples for training: {}".format(nb_samples))
            print("nb samples for validation: {}".format(X_val.shape[1]))
            print("dim input: {}".format(self.dim_input))
            print("dim output: {}".format(self.dim_output))
        batch_size = min(batch_size, nb_samples)
        self.__training_loop(X_train, y_train, X_val, y_val, learning_rate,
            max_epochs, momentum, nb_samples, batch_size,
            early_stopping, splitted_batch_size, verbose)

    def predict(self, X, preprocess=True):
        pred = self.predict_probas(X, preprocess=preprocess)
        labels = [max(p.items(), key=operator.itemgetter(1))[0] for p in pred]
        return labels

    def predict_probas(self, X, preprocess=True):
        if preprocess:
            X = self.__preprocess_test(np.matrix(X))
        raw_predict = self.__predict(X).T
        predict = [{self.labels[k]: raw_predict[n][k] for k in
            range(raw_predict.shape[1])} for n in range(raw_predict.shape[0])]
        return predict

    def predict_mean_probas(self, X, preprocess=True):
        """ Only for numerical labels predictions """
        pred = self.predict_probas(X, preprocess=preprocess)
        mean = [int(sum([float(k) * float(v) for k,v in p.items()])) for p in pred]
        return mean

    @classmethod
    def load(cls, model):
        with open(model, 'rb') as fd:
            if PYTHON_MAJOR == 3:
                ret = pickle.load(fd, encoding="latin1")
            else:
                ret = pickle.load(fd)
        return ret

    def dump(self, model_name=None, directory=None, verbose=False):
        if model_name is None:
            model_name = str(datetime.datetime.now()).split('.')[0]  + ".pkl"
        if directory is None:
            directory = resource_filename(__name__, '../models')
            if not os.path.exists(directory):
                os.mkdir(directory)
        filename = os.path.realpath(os.path.join(directory, model_name))
        self.__clean()
        with open(filename, 'wb') as fd:
            pickle.dump(self, fd, protocol=0)
        if verbose:
            print("model saved in {}".format(filename))

    def __clean(self):
        for layer in self.layers:
            if not layer.is_network_input:
                layer.clean()

    def __training_loop(self, X_train, y_train, X_val, y_val, learning_rate,
                        max_epochs, momentum, nb_samples, batch_size,
                        early_stopping, splitted_batch_size, verbose):
        last_loss, best_loss = float("inf"), float("inf")
        weights_biases = [(l.weights.copy(), l.biases.copy()) for
                          l in self.layers]
        last_improvement = 0
        epoch = 0
        if verbose:
            print("Training starts:")
            val_pred = self.__predict(X_val)
            loss = cross_entropy_loss(val_pred, y_val)
            print("val cross entropy loss at beginning: {}".format(
                                                            round(loss, 5)))
        try:
            while epoch <= max_epochs:
                index = np.random.choice(nb_samples, batch_size, replace=False)
                subsample = X_train[:, index]
                observations = y_train[:, index]
                subsamples = split_batch(subsample, splitted_batch_size)
                _observationss = split_batch(observations, splitted_batch_size)
                for subsample, observations in zip(subsamples, _observationss):
                    predictions = self.__predict(subsample)
                    errors = observations - predictions
                    self.__backprop(errors)
                    self.__get_d_weights_tmp(learning_rate, batch_size)
                self.__weights_update(momentum)
                val_pred = self.__predict(X_val)
                loss = cross_entropy_loss(val_pred, y_val)
                if early_stopping != -1:
                    if loss > best_loss:
                        last_improvement += 1
                    else:
                        weights_biases = [(l.weights.copy(), l.biases.copy()) for
                                          l in self.layers]
                        last_improvement = 0
                        best_loss = loss
                    if last_improvement >= early_stopping:
                        if verbose:
                            print("Early stopping")
                        self.__force_weights_biases(weights_biases)
                        break
                last_loss = loss
                epoch += 1
                if verbose:
                    print("val cross entropy loss after epoch {}: {}".format(
                        epoch, round(loss, 5)))

        except KeyboardInterrupt:
            warnings.warn("Training has been interrupted")
        if early_stopping != -1:
            self.__force_weights_biases(weights_biases)
        if verbose:
            val_pred = self.__predict(X_val)
            loss = cross_entropy_loss(val_pred, y_val)
            print("val cross entropy loss at end: {}".format(round(loss, 5)))

    def __force_weights_biases(self, weights_biases):
        for k, w_b in enumerate(weights_biases):
            self.layers[k].weights, self.layers[k].biases = w_b

    def __backprop(self, errors):
        for n in reversed(range(1, self.nb_layers)):
            layer = self.layers[n]
            if not layer.is_network_output:
                errors = self.layers[n+1].weights.T *\
                         self.layers[n+1].local_grad
            if layer.is_network_output:
                layer.local_grad = errors
            else:
                if layer.activation.name == "tanh":
                    deriv = 1 - layer.neurals ** 2
                elif layer.activation.name in ("logistic", "softmax"):
                    deriv = layer.neurals * (1 - layer.neurals)
                else:
                    deriv = layer.derivation(
                        layer.aggregate(self.layers[n-1].neurals))
                layer.local_grad = np.multiply(errors, deriv)

    def __get_d_weights_tmp(self, learning_rate, batch_size):
        for n in range(1, self.nb_layers):
            layer = self.layers[n]
            layer_1 = self.layers[n - 1]
            d_weights_unit = [grad.T * neural_1.T for grad, neural_1 in
                                  zip(layer.local_grad.T, layer_1.neurals.T)]
            d_weights = learning_rate * sum(d_weights_unit) / batch_size
            mean_local_grad = layer.local_grad.mean(axis=1)
            d_biases = learning_rate * mean_local_grad
            layer.d_weights_tmp.append(d_weights)
            layer.d_biases_tmp.append(d_biases)

    def __weights_update(self, momentum):
        for n in range(1, self.nb_layers):
            layer = self.layers[n]
            layer.d_weights = sum(layer.d_weights_tmp) + momentum * layer.d_weights
            layer.d_biases = sum(layer.d_biases_tmp) + momentum * layer.d_biases
            layer.weights += layer.d_weights
            layer.biases += layer.d_biases
            layer.d_weights_tmp = []
            layer.d_biases_tmp = []

    def __predict(self, X):
        if X.shape[1] == 0:
            return None
        for k, layer in enumerate(self.layers):
            layer.eval(X)
            X = layer.neurals
        return X

    def __init_layers(self, activation):
        if self.hidden_layer_sizes is None:
            self.hidden_layer_sizes = (int(self.dim_input),)
        self.nb_layers = len(self.hidden_layer_sizes) + 2
        self.layers = [Layer(self.dim_input, 0,
                       dtype=self.dtype, is_network_input=True)]
        for hidden_layer_size in self.hidden_layer_sizes:
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

    def __divide(self, X):
        return X / self.divider

    def __remove_ct_features(self, X):
        return X[self.not_ct_features, :]

    def __set_not_ct_features(self, X):
        self.not_ct_features = []
        for i, x in enumerate(X):
            if np.std(x) != 0:
                self.not_ct_features.append(i)

    def __preprocess_test(self, X):
        X.astype(self.dtype, copy=False)
        X = self.__divide(X)
        if self.rm_ct_features:
            X = self.__remove_ct_features(X)
        return self.__standardize(X)

    def __preprocess_train(self, X, y, standardize=True, divider=1.):
        y = np.array(y)
        labels = np.unique(y)
        y_preprocessed = np.matrix(
            [[1 if e == label else 0 for label in labels] for e in y.T]
        ).T
        self.labels = labels
        self.divider = divider
        X = np.matrix(X)
        X.astype(self.dtype, copy=False)
        X_preprocessed = self.__divide(X)
        self.__set_not_ct_features(X)
        if self.rm_ct_features:
            X_preprocessed = self.__remove_ct_features(X_preprocessed)
        if standardize:
            self.mu = np.array([])
            self.sigma = np.array([])
            for x in X_preprocessed:
                self.mu = np.append(self.mu, np.mean(x))
                self.sigma = np.append(self.sigma, np.std(x))
            X_preprocessed = self.__standardize(X_preprocessed)
        else:
            self.mu = np.zeros(self.dim_input)
            self.sigma = np.ones(self.dim_input)
        y_preprocessed.astype(self.dtype, copy=False)
        return X_preprocessed, y_preprocessed

    def __get_dim_input(self, X, rm_ct_features=True):
        X = np.matrix(X)
        dim_input = X.shape[0]
        if rm_ct_features:
            nb_ct_features = 0
            for x in X:
                if np.std(x) == 0:
                    nb_ct_features += 1
            dim_input -= nb_ct_features

        return dim_input

    def __get_dim_output(self, y):
        return np.unique(np.asarray(y)).shape[0]
