#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
import json
import numpy as np
from src import dataframe
from src.math import softmax
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

def transform_label(Y):
    existings = list(set(Y))
    return [[1 if y == existing else 0 for existing in existings] for y in Y]

def main():
    args = parse_arguments()
    df = dataframe.read_csv(args.csvfile)
    df.get_numerical_features()
    df.digitalize()
    df.replace_nan()
    df.standardize()
    X = np.array([x for feature, x in df.standardized.items() if feature in df.numerical_features and
                                        feature != "Index"])
    X = X.T
    np.random.shuffle(X)
    #X = np.insert(X, 0,  [1 for _ in X[0]], axis=0)
    #for x in X:
    #    print(x)
    #return 
    #Y = transform_label(df.standardized['diagnosis'])
    Y = transform_label(df.standardized['Hogwarts House'])
    dim_input, dim_output = len(X[0]), len(Y[0])
    mlp = Mlp(dim_input, dim_output)
    
    for n in range(50):
        predict = mlp.predict(X[n])
        print("predict n : ", predict, "    real:", Y[n])
    return 

    mlp.print_weights()
    mlp.fit(X, Y)
    print("cost:", mlp.cost(X, Y))
    #for n in range(len(X)):
    for n in range(10):
        #print(X[n])
        predict = mlp.predict(X[n])
        print("predict n : ", predict, "    real:", Y[n])
    mlp.print_weights()
    print("sum(predict):", sum(predict))

if __name__ == '__main__':
    main()
