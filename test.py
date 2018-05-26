#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
import json
import numpy as np
from src import dataframe
from src.activations import Relu
from src.layer import Layer
from src.mlp import Mlp

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='dataset_train.csv')
    args = parser.parse_args()

    return args

def transform_label(Y, label):
    return [1 if y == label else 0 for y in Y]

def main():
    args = parse_arguments()
    df = dataframe.read_csv(args.csvfile)
    df.get_numerical_features()
    df.digitalize()
    df.replace_nan()
    df.standardize()
    X = np.array([x for feature, x in df.standardized.items() if feature in df.numerical_features])
    X = X.T
    Y = transform_label(df.standardized['diagnosis'], 'M')
    print(Y)
    relu = Relu()
    layer = Layer(3, 3)
    layer.init()
    mlp = Mlp()
    mlp.fit(X, Y)
    predict = mlp.predict(X[0])
    print("predict:", predict)
    print("sum(predict):", sum(predict))

if __name__ == '__main__':
    main()
