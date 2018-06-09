import numpy as np
from src.layer import Layer
from src.utils import get_random_index

DEBUG = True
DEBUG = False

class Mlp(object):
    def __init__(self, hidden_layer_sizes=(1, 5), activation="relu"):
        self.nb_hidden_layers = hidden_layer_sizes[0]
        self.nb_neurals_by_hidden_layer = hidden_layer_sizes[1]
        self.nb_layers = self.nb_hidden_layers + 2
        self.__init_layers__(activation)

    def __init_layers__(self, activation):
        self.layers = [Layer(
            self.nb_neurals_by_hidden_layer,
            self.nb_neurals_by_hidden_layer,
            activation=activation
        ) for _ in range(self.nb_hidden_layers)]
        [layer.init() for layer in self.layers]

    def fit(self, X, Y, learning_rate=1e-2, batch_size=1, epochs=5000, momentum=0.5):
        self.layers = [Layer(self.nb_neurals_by_hidden_layer, X.shape[1])] + self.layers
        self.layers.append(Layer(
            len(Y[0]), self.nb_neurals_by_hidden_layer,
            activation="softmax", output_layer=True)
        )
        self.layers[0].init()
        self.layers[-1].init()
        #batch_size = len(X)
        i = 0
        epoch = 0
        while epoch < epochs:
            #index = get_random_index(len(X), batch_size)
            index = [i]
            predictions = [list(self.predict(X[i])) for i in index]
            observations = [Y[i] for i in index]
            errors = [sum([predictions[i][k] - observations[i][k] for i in range(len(predictions))]) \
                        for k in range(len(predictions[0]))]
            self.get_local_gradients(errors)
            self.update_weights(learning_rate, momentum)
            if epoch % 100 == 0:
                print("cost at epoch {}: {}", epoch, self.cost(X, Y))
                print(self.predict(X[0]))
            epoch += 1
            i = (i + 1) % len(X)

    def get_local_gradients(self, errors):
        #print("begin get_local_gradients")
        for n in reversed(range(self.nb_layers)):
            layer = self.layers[n]
            if DEBUG:
                print("n:", n, "neurals:", layer.neurals)
            if n == self.nb_layers - 1:
                layer.local_gradients = errors *\
                    layer.activation.get_derivative(layer.neurals)
            else:
                layer.local_gradients = layer.activation.get_derivative(layer.neurals) *\
                    np.array([sum([self.layers[n+1].local_gradients[k] * self.layers[n+1].weights[k][j] for\
                        k in range(len(self.layers[n+1].local_gradients))]) for j in range(len(layer.neurals))])
            if DEBUG:
                print("n:", n, "len(local_gradients):", len(layer.local_gradients))
        #print("end get_local_gradients")

    def update_weights(self, learning_rate, momentum):
        if DEBUG:
            print("begin update_weights")
        for n in range(1, self.nb_layers):
            #layer = self.layers[n]
            if DEBUG:
                print("n in update_weights:", n)
            for j in range(len(self.layers[n].weights)):
                if DEBUG:
                    print("j in update_weights:", j)
                for i in range(len(self.layers[n].weights[j])):
                    if DEBUG:
                        print("i in update_weights:", i)
                        print("local_gradients:", self.layers[n].local_gradients)
                    #print("self.layers[{}].weights".format(n), self.layers[n].weights)
                    neurals_n_1 = self.layers[n-1].neurals
                    neurals_n_1 = np.insert(neurals_n_1, 0, -0.5, axis=0)
                    self.layers[n].deltas[j][i] = learning_rate * self.layers[n].local_gradients[j] *\
                        neurals_n_1[i] + (self.layers[n].deltas[j][i] * momentum)
                    self.layers[n].weights[j][i] += self.layers[n].deltas[j][i]
            #print(self.layers[n].deltas)
        #print("hello")
                    #self.layers[n].weights[j][i] += learning_rate *\
                    #self.layers[n].local_gradients[j] * neurals_n_1[i]
        if DEBUG:
            print("end update_weights")

    def predict(self, X):
        X = X.T
        for k, layer in enumerate(self.layers):
            if DEBUG:
                print("k:", k)
            layer.eval(X)
            X = layer.neurals
            if DEBUG:
                print(X)

        if DEBUG:
            print()
       # print("end predict")
        return X

    def cost(self, X, Y):
        return sum([(1. / 2) *sum([a ** 2 for a in (self.predict(x) - y)]) for
                            x, y in zip(X, Y)])

    def print_weights(self):
        for k, layer in enumerate(self.layers):
            print("weights in layer {}".format(k), layer.weights)
