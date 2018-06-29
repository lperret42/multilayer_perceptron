#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
import json
import numpy as np
from src import dataframe
from src.math import softmax
from src.activations import Relu
from src.layer import Layer
from src.mlp import Mlp
from src.utils import get_randomized_data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    output_label = 'diagnosis'     # diagnosis, iris, Hogwarts House
    output_label = 'iris'
    output_label = 'Hogwarts House'
    output_label = 'age'
    args = parse_arguments()
    df = dataframe.read_csv(args.csvfile)
    df.get_numerical_features()
    df.digitalize()
    df.replace_nan()
    df.standardize()
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    Y = df.data[output_label]
    dim_input, dim_output = len(X), len(list(set(Y)))
    mlp = Mlp(dim_input, dim_output, hidden_layer_sizes=(128, 128, 128, 128, 128, 100))
    print("before fit")
    """
    X_train, Y_train, X_test, Y_test = mlp.train_test_split(X,  Y)
    print("X.shape", (len(X), len(X[0])))
    print("Y.shape", len(Y))
    print("X_train.shape", X_train.shape)
    print("Y_train.shape", Y_train.shape)
    mlp.fit(X_train, Y_train, verbose=True)
    """
    mlp.fit(X, Y, verbose=True)
    predictions = mlp.predict_labels(X)
    [print("predict: {}   real: {}".format(
        pred, obs)) for pred, obs in zip(predictions, Y)]
    print("precision:", mlp.get_precision(predictions, Y))
    print("mean error:", mlp.get_mean_error(predictions, Y))
    return

if __name__ == '__main__':
    main()
