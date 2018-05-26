from src.utils import is_int, quicksort
from math import exp, log

def abs(x):
    return -x if x < 0 else x

def min(lst):
    m = lst[0]
    for elem in lst:
        if elem < m:
            m = elem
    return m

def max(lst):
    m = lst[0]
    for elem in lst:
        if elem > m:
            m = elem
    return m

def sum(lst):
    s = 0
    for elem in lst:
        s += elem
    return s

def mean(lst):
    return sum(lst) / len(lst)

def sum_with_empty(lst):
    s = 0
    for elem in lst:
        s += elem if elem != "" else 0
    return s

def mean_with_empty(lst):
    return sum_with_empty(lst) / len([elem for elem in lst if elem != ""])

def std(lst):
    mu = mean(lst)
    return sqrt(sum([(x - mu) ** 2 for x in lst]) / (len(lst) - 1))

def ceil(f):
    return int(f) if is_int(f) or f < 0 else int(f) + 1

def quartile_n(lst, n):
    if n == 2:
        if len(lst) % 2 == 0:
            return mean([lst[int(len(lst) / 2) -1] , lst[int(len(lst) / 2)]])
        else:
            return lst[int(len(lst) / 2)]
    else:
        return lst[ceil(float(n / 4.) * len(lst))]

def scalar_product(X, Y):
    return sum([x * y for x, y in zip(X, Y)])

def linear_function(a, b, x):
    return a * x + b

def logistic_function(x):
    return 1. / (1 + exp(-x)) if -x <= 500 else 0

def h_theta(THETA, X):
    return min([logistic_function(scalar_product(THETA, X)), 0.99999])

def logistic_cost(THETA, X, Y):
    return (- 1. / len(Y)) * sum([y * log(h_theta(THETA, x)) + ((1 - y) *
        log(1 - h_theta(THETA, x))) for y, x in zip(Y, X)])

def partial_derivative_n(THETA, X, Y, n):
    return (1. / len(Y)) * sum([(h_theta(THETA, x) - y) * x[n] for y, x in zip(Y, X)])

def sqrt(x, epsilon=10e-15):
    """
    implementation of the sqrt function, both suites u and v converge to sqrt(x)
    """
    if x < 0:
        return None
    if x == 0:
        return 0
    u = 1
    v = x
    error_u = abs(u * u - x)
    error_v = abs(v * v - x)
    old_error_u = error_u
    old_error_v = error_v
    while error_u > epsilon and error_v > epsilon:
        tmp = u
        u = 2. / (1. / u + 1. / v)
        v = (tmp + v) / 2.
        error_u = abs(u * u - x)
        error_v = abs(v * v - x)
        if old_error_u == error_u and old_error_v == old_error_v:
            break
        old_error_u = error_u
        old_error_v = error_v

    return u if error_u <= error_v else v

def logistic(x):
    return 1. / (1 + exp(-x)) if -x <= 500 else 0

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
