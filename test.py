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
    parser.add_argument('csvfile', help='dataset_train.csv')
    args = parser.parse_args()

    return args

def transform_label(Y):
    existings = list(set(Y))
    print(existings)
    return [np.array([1 if y == existing else 0 for existing in existings]) for y in Y]

def main():
    output_label = 'diagnosis'     # diagnosis, iris, Hogwarts House
    args = parse_arguments()
    df = dataframe.read_csv(args.csvfile)
    df.get_numerical_features()
    df.digitalize()
    df.replace_nan()
    df.standardize()
    #X = np.array([x for feature, x in df.standardized.items() if feature in df.numerical_features])
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    #Y = df.standardized[output_label]
    Y = df.data[output_label]
    dim_input, dim_output = len(X), len(list(set(Y)))
    mlp = Mlp(dim_input, dim_output)
    mlp.fit(X, Y)
    X = mlp.standardize(np.matrix(X))
    predictions = mlp.predict_labels(X)
    print("precision:", mlp.get_precision(predictions, Y))
    return
    #X, Y = get_randomized_data(X, Y)
    #index = range(len(Y))
    #samples = [np.matrix(X[i]).T for i in index]
    #observations = [np.matrix([Y[i] for i in index]).T]
    #for n in range(len(Y)):
    print(Y.shape)
    for n in Y.shape[1]:
        #predict = mlp.predict(samples[n])
        predict = mlp.predict(X[n])
        #print("predict n : ", [round(p, 3) for p in predict], "    real:", Y[n])
        print("predict n : ", predict, "    real:", Y[n])
    #print("precision:", mlp.get_precision(samples, observations))

if __name__ == '__main__':
    main()
