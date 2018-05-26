from src.math import relu, d_relu, logistic, d_logistic, tanh, d_tanh

class Activation(object):
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def get_function(self, x):
        return self.function(x)

    def get_derivative(self, x):
        return self.derivative(x)

class Relu(Activation):
    def __init__(self):
        self.function = relu
        self.derivative = d_relu

class Logistic(Activation):
    def __init__(self):
        self.function = logistic
        self.derivative = d_logistic

class Tanh(Activation):
    def __init__(self):
        self.function = tanh
        self.derivative = d_tanh
