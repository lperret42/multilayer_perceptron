#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
import json
import numpy as np
from src import dataframe
from src.math import softmax
from src.activations import Relu
from src.layer import Layer
from src.mlp import Mlp
from sklearn.neural_network import MLPClassifier
from src.utils import train_test_split, prediction_precision,\
                      prediction_mean_error, mean_squared_error,\
                      cross_entropy_loss

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    output_label = 'Hogwarts House'
    output_label = 'age'
    output_label = 'iris'
    output_label = 'diagnosis'
    args = parse_arguments()
    df = dataframe.DataFrame.read_csv(args.csvfile)
    #df.set_numerical_features(to_remove=["age"])
    df.set_numerical_features(to_remove=["id"])
    df.digitalize()
    df.replace_nan()
    df.standardize()
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    Y = df.data[output_label]
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, train_ratio=0.8)
    clf = MLPClassifier(
        verbose=True,
        #max_iter=2000,
        #tol=0.00000000001,
        #hidden_layer_sizes=(16,)
        )
    print(clf)
    X_train = np.array(X_train).T.tolist()
    Y_train = np.array(Y_train).T.tolist()
    X_test = np.array(X_test).T.tolist()
    Y_test = np.array(Y_test).T.tolist()
    print("X_train.shape:", len(X_train), len(X_train[0]))
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    predictions = np.matrix(predictions).T
    Y_test = [y[0] for y in Y_test]
    [print("predict: {}   real: {}".format(
        pred, obs)) for pred, obs in zip(predictions, Y_test)]
    print("precision:", prediction_precision(predictions, Y_test))
    print("mean error:", prediction_mean_error(predictions, Y_test))
    print("squared error:", mean_squared_error(predictions, Y_test))
    print(cross_entropy_loss(predictions, Y_test))
    return

if __name__ == '__main__':
    main()
