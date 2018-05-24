import matplotlib.pyplot as plt
import time
import csv

from src.math import logistic_function, logistic_cost, partial_derivative_n

class LogisticRegressor(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.ndims = len(X[0])
        self.theta = [0 for _ in range(self.ndims)]

    def update_params(self, learning_rate):
        theta = [learning_rate * partial_derivative_n(self.theta, self.X, self.Y, n) for
                                    n in range(self.ndims)]
        self.theta = [self.theta[i] - theta[i] for i in range(self.ndims)]

    def train(self, learning_rate=1, max_iter=2000, epsilon=1e-6, print_cost=False):
        epoch = 0
        while True:
            cost = logistic_cost(self.theta, self.X, self.Y)
            if print_cost and epoch % 100 == 0:
                print(epoch, "/", max_iter, ": loss ", cost, sep="")
            if epoch >= max_iter:
                break
            self.update_params(learning_rate)
            epoch += 1
