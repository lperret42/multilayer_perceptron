import numpy as np
from src.layer import Layer
from src.utils import get_random_index

DEBUG = True
DEBUG = False

class Mlp(object):
    #def __init__(self, dim_input, dim_output, hidden_layer_sizes=[4, 4], activation="logistic"):
    def __init__(self, dim_input, dim_output, hidden_layer_sizes=[], activation="logistic"):
        self.nb_hidden_layers = len(hidden_layer_sizes)
        self.nb_layers = self.nb_hidden_layers + 2
        self.__init_layers__(dim_input, dim_output, activation, hidden_layer_sizes)

    def __init_layers__(self, dim_input, dim_output, activation, hidden_layer_sizes):
        self.layers = [Layer(dim_input, 0, activation=activation, input_layer=True)]
        for hidden_layer_size in hidden_layer_sizes:
            self.layers += [Layer(
                hidden_layer_size,
                self.layers[-1].nb_neural,
                activation=activation,
            )]
            self.layers[-1].init()
        self.layers += [Layer(
            dim_output,
            self.layers[-1].nb_neural,
            activation="softmax",
            output_layer=True,
        )]
        self.layers[-1].init()

    def fit(self, X, Y, learning_rate=1e-2, batch_size=8, epochs=10000, momentum=0.75):
        self.print_layers()
        i = 0
        epoch = 0
        while epoch < epochs:
            #print("*************** new epoch *******************")
            index = get_random_index(len(X), batch_size)
            predictions = [list(self.predict(X[i])) for i in index]
            observations = [Y[i] for i in index]
            #self.show_one_result(X[index[0]], Y[index[0]])
            if DEBUG:
                print("predictions:", predictions)
                print("observations:", observations)
            errors = [sum([observations[i][k] - predictions[i][k] for i in range(len(predictions))]) \
                        for k in range(len(predictions[0]))]
            #print("X:", X[index[0]])
            #print("predict:", self.predict(X[index[0]]))
            #print("Y:", Y[index[0]])
            #print("errors: ", errors)
            self.get_local_gradients(errors)
            self.update_weights(learning_rate, momentum)
            if epoch % 100 == 0:
                print("cost at epoch {}: {}".format(epoch, self.cost(X, Y)))
                print(self.predict(X[0]))
            epoch += 1
            i = (i + 1) % len(X)
            #self.show_one_result(X[index[0]], Y[index[0]])

    def get_local_gradients(self, errors):
        #print("begin get_local_gradients")
        for n in reversed(range(1, self.nb_layers)):
            layer = self.layers[n]
            if DEBUG:
                print("n:", n, "neurals:", layer.neurals)
                print("n:", n, "weights:", layer.weights)
                print("n:", n, "errors:", errors)
                print("n:", n, "layer.activation.get_derivative(layer.neurals",
                        layer.activation.get_derivative(layer.neurals))
            if n == self.nb_layers - 1:
                layer.local_gradients = errors * (layer.neurals * (1 - layer.neurals))
            else:
                #layer.local_gradients = layer.activation.get_derivative(layer.neurals[1:]) *\
                layer.local_gradients = layer.neurals[1:] * (1 - layer.neurals[1:]) *\
                    np.array([sum([self.layers[n+1].local_gradients[k] * self.layers[n+1].weights[k][j] for\
                    k in range(len(self.layers[n+1].local_gradients))]) for j in range(len(layer.neurals[1:]))])
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
                    """
                    neurals_n_1 = self.layers[n-1].neurals
                    self.layers[n].deltas[j][i] = learning_rate * self.layers[n].local_gradients[j] *\
                        neurals_n_1[i] + (self.layers[n].deltas[j][i] * momentum)
                    self.layers[n].weights[j][i] -= self.layers[n].deltas[j][i]
                    """
                    self.layers[n].deltas[j][i] = learning_rate * self.layers[n].local_gradients[j] *\
                        self.layers[n-1].neurals[i]
                    self.layers[n].weights[j][i] += self.layers[n].deltas[j][i]

        if DEBUG:
            print("end update_weights")

    def predict(self, x):
        if DEBUG:
            print("***************Predict***********************")
            print("x initial:", x)
        x = x.T
        for k, layer in enumerate(self.layers):
            if DEBUG:
                print("k:", k)
            layer.eval(x)
            x = layer.neurals
            if DEBUG:
                print("x after layer {}:".format(k), x)

        if DEBUG:
            print()
       # print("end predict")
        return x

    def cost(self, X, Y):
        return sum([(1. / 2) * sum([a ** 2 for a in (self.predict(x) - y)]) for
                            x, y in zip(X, Y)])

    def show_one_result(self, x, y):
        prediction = self.predict(x)
        print("prediction: {}     real: {}".format(prediction, y))

    def print_weights(self):
        for k, layer in enumerate(self.layers):
            print("weights in layer {}".format(k), layer.weights)

    def print_layers(self):
        for k, layer in enumerate(self.layers):
            print("layer {}:\n".format(k), layer)
