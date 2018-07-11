import csv
from math import log
import numpy as np

def train_test_split(X, y, train_ratio=0.8):
    X = np.matrix(X)
    y = np.matrix(y)
    nb_samples = X.shape[1]
    nb_for_train = int(train_ratio * nb_samples)
    index_train = np.random.choice(nb_samples, nb_for_train, replace=False)
    index_test = list(set(range(nb_samples)) -set(index_train))
    X_train, X_test = X[:, index_train], X[:, index_test]
    y_train, y_test = y[:, index_train], y[:, index_test]
    return X_train, y_train, X_test, y_test

def mean_squared_error(predictions, observations):
    errors = observations - predictions
    return float((1 / errors.shape[1]) * sum([e.dot(e.T) for e in errors.T]))

def cross_entropy_loss(predictions, observations, margin=1e-20):
    np.clip(predictions, margin, 1 - margin, out=predictions)
    log_pred = np.vectorize(log)(predictions)
    log_pred_comp = np.vectorize(log)(1-predictions)
    product = np.multiply(observations, log_pred)
    return (-product.sum(axis=0)).mean()

def get_uniform_matrix(sizes, low=0, high=1):
    if isinstance(sizes, int) or isinstance(sizes, float):
        dim_one = True
        sizes = (sizes, 1)
    elif len(sizes) == 1:
        sizes = (sizes[0], 1)
        dim_one = True
    else:
        dim_one = False
    weights = np.random.uniform(low, high, sizes)
    if dim_one:
        weights = np.matrix(weights)
    return weights

def get_normal_matrix(sizes, mu=0, sigma=1):
    if isinstance(sizes, int) or isinstance(sizes, float):
        dim_one = True
        sizes = (sizes, 1)
    elif len(sizes) == 1:
        dim_one = True
        sizes = (sizes[0], 1)
    else:
        dim_one = False
    weights = np.random.uniform(mu, sigma, sizes)
    if dim_one:
        weights = np.matrix(weights)
    return weights

def print_pred_vs_obs(predictions, observations):
    [print("predict: {:>12}    real: {:>12}".format(
        pred, obs)) for pred, obs in zip(predictions, observations)]

def pred_accuracy(predictions, observations):
    return sum([1 if pred == obs else 0 for pred, obs in
        zip(predictions, observations)]) / len(predictions)

def pred_mean_error(predictions, observations):
    if len(predictions) == 0 or not is_float(predictions[0]):
        return None
    return round(np.mean([abs(pred - obs) for pred, obs in
        zip(predictions, observations)]), 3)

def sum_with_empty(lst):
    s = 0
    for elem in lst:
        s += elem if elem != "" else 0
    return s

def mean_with_empty(lst):
    return sum_with_empty(lst) / len([elem for elem in lst if elem != ""])

def get_data(csv_file):
    with open(csv_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')]
        csvfile.close()

    data = {}
    features = lines[0]
    for feature in features:
        data[feature] = []
    lines.pop(0)
    for line in lines:
        for i, value in enumerate(line):
            data[features[i]].append(value)
    return data

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_list_num(lst):
    for elem in lst:
        if elem != "" and not is_float(elem):
            return False
    return True

def quicksort(lst, rev=False):
    if len(lst) < 2:
        return lst
    else:
        pivot = lst[0] if rev == False else lst[-1]
        less = [i for i in lst[1:] if i <= pivot] if rev == False\
                            else [i for i in lst[:-1] if i <= pivot]
        greater = [i for i in lst[1:] if i > pivot] if rev == False\
                            else [i for i in lst[:-1] if i > pivot]
        return quicksort(less, rev=rev) + [pivot] +\
            quicksort(greater, rev=rev) if rev == False\
            else quicksort(greater, rev=rev) + [pivot] + quicksort(less, rev=rev)

def keep_only_float(lst):
    ret = []
    for elem in lst:
        if is_float(elem):
            ret.append(float(elem))
    return ret
