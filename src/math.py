import numpy as np
from math import exp, log

def logistic(x):
    return 1. / (1 + exp(-x))

def d_logistic(x):
    tmp = logistic(x)
    return tmp * (1 - tmp)

def relu(x):
    return 0 if x < 0 else x

def d_relu(x):
    return 0 if x < 0 else 1

def tanh(x):
    return 2. / (1 + exp(-2 * x)) -1

def d_tanh(x):
    return 1. - tanh(x) ** 2

def softmax(X):
    exps = [exp(x) for x in X]
    return np.array([x / sum(exps) for x in exps])

def d_softmax(X):
    tmp = softmax(X)
    return np.array(tmp * (1 - tmp))
