import argparse
import json
import numpy as np
from src import dataframe
from src.mlp import Mlp
from sklearn.neural_network import MLPClassifier
from src.utils import train_test_split, print_pred_vs_obs,\
                      prediction_precision, prediction_mean_error,\
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
    y = df.data[output_label]
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.8)
    unique = np.unique(np.asarray(y_train))
    dim_input, dim_output = X_train.shape[0], np.unique(np.asarray(y_train)).shape[0]
    #mlp = Mlp(dim_input, dim_output, hidden_layer_sizes=(50, 100, 12, 132, 90,))
    #mlp = Mlp(dim_input, dim_output)
    mlp = Mlp(dim_input, dim_output, hidden_layer_sizes=(100,))
    mlp.fit(X_train, y_train, early_stopping=True, verbose=True)
    #mlp.fit(X_train, y_train, early_stopping=True)
    predictions = mlp.predict_probas(X_test)
    y_test_vector = [y[0] for y in y_test.T.tolist()]
    print_pred_vs_obs(predictions, y_test_vector)
    print("precision:", prediction_precision(predictions, y_test.T))
    print("mean error:", prediction_mean_error(predictions, y_test.T))
    return

if __name__ == '__main__':
    main()
