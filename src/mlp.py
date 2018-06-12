import numpy as np
from src.layer import Layer
from src.utils import get_random_index, get_randomized_data

class Mlp(object):
    def __init__(self, dim_input, dim_output, hidden_layer_sizes=[4], activation="tanh"):
        self.nb_hidden_layers = len(hidden_layer_sizes)
        self.nb_layers = self.nb_hidden_layers + 2
        self.__init_layers__(dim_input, dim_output, activation, hidden_layer_sizes)

    def __init_layers__(self, dim_input, dim_output, activation, hidden_layer_sizes):
        self.layers = [Layer(dim_input, 0, activation=activation, input_layer=True)]
        for hidden_layer_size in hidden_layer_sizes:
            self.layers += [Layer(
                hidden_layer_size,
                self.layers[-1].size,
                activation=activation,
            )]
        self.layers += [Layer(
            dim_output,
            self.layers[-1].size,
            activation="softmax",
            output_layer=True,
        )]

    def fit(self, X, Y, learning_rate=0.1, batch_size=1, epochs=20000, momentum=0.5):
        X, Y = get_randomized_data(X, Y)
        i = 0
        epoch = 0
        while epoch < epochs:
            index = get_random_index(len(X), batch_size)
            index = [i]
            predictions = [list(self.predict(X[i])) for i in index]
            observations = [Y[i] for i in index]
            errors = [np.mean([observations[i][k] - predictions[i][k] for i in
                range(len(predictions))]) for k in range(len(predictions[0]))]
            self.get_local_gradients(errors)
            self.update_weights(learning_rate, momentum)
            if epoch % 100 == 0:
                print("cost at epoch {}: {}".format(epoch, self.cost(X, Y)))
            epoch += 1
            i = (i + 1) % len(X)

    def get_local_gradients(self, errors):
        for n in reversed(range(1, self.nb_layers)):
            layer = self.layers[n]
            layer_n_1 = self.layers[n - 1]
            if n == self.nb_layers - 1:
                deriv = np.array(layer.activation.deriv(layer.weights.dot(layer_n_1.neurals) +
                    layer.biases))
                layer.local_gradients = errors * deriv
            else:
                deriv = np.array(layer.activation.deriv(layer.weights.dot(layer_n_1.neurals) +
                    layer.biases))
                errors = self.layers[n+1].weights.T.dot(self.layers[n+1].local_gradients)
                layer.local_gradients = errors * deriv

    def update_weights(self, learning_rate, momentum):
        for n in range(1, self.nb_layers):
            layer = self.layers[n]
            layer_1 = self.layers[n - 1]
            for j in range(len(layer.weights)):
                for i in range(len(layer.weights[j])):
                    layer.deltas[j][i] = learning_rate * layer.local_gradients[j] *\
                        layer_1.neurals[i] + momentum * layer.deltas[j][i]
                    layer.weights[j][i] += layer.deltas[j][i]

            layer.biases += learning_rate * layer.local_gradients

    def predict(self, x):
        x = x.T
        for k, layer in enumerate(self.layers):
            layer.eval(x)
            x = layer.neurals
        return x

    def predict_label(self, x):
        predict = list(self.predict(x))
        return predict.index(max(predict))

    def cost(self, X, Y):
        return sum([(1. / 2) * sum([a ** 2 for a in (self.predict(x) - y)]) for
                            x, y in zip(X, Y)])

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
            y = Y[i]
            predict_label = self.predict_label(x)
            real_label = y.index(max(y))
            if predict_label != real_label:
                nb_errors += 1
        return 1 - nb_errors / len(X)
