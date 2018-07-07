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
from src.utils import get_randomized_data, train_test_split,\
                      multi_to_one

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    output_label = 'age'
    #output_label = 'diagnosis'
    args = parse_arguments()
    df = dataframe.read_csv(args.csvfile)
    #df.get_numerical_features(remove=["id"])
    df.get_numerical_features(remove=["age"])
    df.digitalize()
    df.replace_nan()
    df.standardize()
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    Y = df.data[output_label]
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, train_ratio=0.8)
    unique = np.unique(np.asarray(Y_train))
    dim_input, dim_output = X_train.shape[0], np.unique(np.asarray(Y_train)).shape[0]
    mlp = Mlp(dim_input, dim_output, hidden_layer_sizes=
        (100,),
    )
    mlp.fit(X_train, Y_train, verbose=True)
    predictions = mlp.predict_labels(X_test)
    Y_test_vector = [y[0] for y in Y_test.T.tolist()]
    mlp.print_pred_vs_obs(predictions, Y_test_vector)
    print("precision:", mlp.get_precision(predictions, Y_test.T))
    print("mean error:", mlp.get_mean_error(predictions, Y_test.T))
    return

if __name__ == '__main__':
    main()
