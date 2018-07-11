import numpy as np
from math import exp

def logistic(x):
    return 1. / (1 + exp(-x))

logistic_vec = np.vectorize(logistic)

def d_logistic(x):
    tmp = logistic(x)
    return tmp * (1 - tmp)

d_logistic_vec = np.vectorize(d_logistic)

def relu(x):
    return 0 if x < 0 else x

relu_vec = np.vectorize(relu)

def d_relu(x):
    return 0 if x < 0 else 1

d_relu_vec = np.vectorize(d_relu)

def tanh(x):
    return 2. / (1 + exp(-2 * x)) -1

tanh_vec = np.vectorize(tanh)

def d_tanh(x):
    return 1. - tanh(x) ** 2

d_tanh_vec = np.vectorize(d_tanh)

exp_vec = np.vectorize(exp)

def softmax(X):
    X = np.asarray(X)
    exps = exp_vec(X)
    sum_exps = np.sum(exps)
    return [x / sum_exps for x in exps]

def d_softmax(X):
    tmp = np.array(softmax(X))
    return tmp * (1 - tmp)
