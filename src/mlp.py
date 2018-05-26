import numpy as np
from src.layer import Layer

class Mlp(object):
    def __init__(self, hidden_layer_sizes=(3, 5), activation="relu"):
        self.nb_layers = hidden_layer_sizes[0]
        self.nb_neurals_by_hidden_layer = hidden_layer_sizes[1]
        self.__init__layers(activation)

    def __init__layers(self, activation):
        self.layers = [Layer(
            self.nb_neurals_by_hidden_layer,
            self.nb_neurals_by_hidden_layer,
            activation=activation
        ) for _ in range(self.nb_layers)]
        [layer.init() for layer in self.layers]

    def fit(self, X, Y, epochs=100):
        self.layers[0] = Layer(self.nb_neurals_by_hidden_layer, X.shape[1])
        self.layers.append(Layer(len(list(set(Y))), self.nb_neurals_by_hidden_layer, output_layer=True))
        self.layers[0].init()
        self.layers[-1].init()

        i = 0
        epoch = 0
        while epoch < epochs:
            predictions = self.predict(X[i])
            errors = Y[i] - predictions
            print(errors)
            epoch += 1
            i = (i + 1) % (X.shape[1])

    def predict(self, X):
        X = X.T
        for k, layer in enumerate(self.layers):
            X = layer.eval(X)
        return X
