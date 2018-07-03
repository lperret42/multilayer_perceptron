import csv
import numpy as np

def train_test_split(X, Y, train_ratio=0.8):
    X = np.matrix(X)
    Y = np.matrix(Y)
    nb_samples = X.shape[1]
    nb_for_train = int(train_ratio * nb_samples)
    index_train = get_random_index(nb_samples, nb_for_train)
    index_test = [i for i in range(nb_samples) if i not in index_train]
    X_train, X_test = X[:, index_train], X[:, index_test]
    Y_train, Y_test = Y[:, index_train], Y[:, index_test]
    return X_train, Y_train, X_test, Y_test


def get_random_index(nb_elem, batch_size):
    index = list(range(nb_elem))
    np.random.shuffle(index)
    return index[:batch_size]

def get_randomized_data(X, Y):
    if len(X) != len(Y):
        raise Exception("nb input and nb output are not equals !")
    index = get_random_index(len(X), len(Y))
    X_randomized = [X[i] for i in index]
    Y_randomized = [Y[i] for i in index]
    return X_randomized, Y_randomized

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
        return quicksort(less, rev=rev) + [pivot] + quicksort(greater, rev=rev) if rev == False\
                else quicksort(greater, rev=rev) + [pivot] + quicksort(less, rev=rev)

def keep_only_float(lst):
    ret = []
    for elem in lst:
        if is_float(elem):
            ret.append(float(elem))
    return ret

def is_int(f):
    return int(f) == f
