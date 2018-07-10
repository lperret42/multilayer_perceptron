import argparse
import json
import numpy as np
from src import dataframe
from src.mlp import Mlp
from src.utils import train_test_split, print_pred_vs_obs,\
                      prediction_precision, prediction_mean_error

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='data.csv')
    parser.add_argument('model', help='model.pkl')
    args = parser.parse_args()

    return args

def main():
    output_label = 'diagnosis'
    output_label = 'iris'
    output_label = 'age'
    args = parse_arguments()
    df = dataframe.DataFrame.read_csv(args.csvfile)
    df.set_numerical_features(to_remove=["age"])
    df.digitalize()
    print("after digitalize")
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    y = df.data[output_label]
    X_test, y_test, _, _ = train_test_split(X, y, train_ratio=0.1)
    mlp = Mlp.load(args.model)
    probas = mlp.predict_probas(X_test)
    predictions = mlp.predict(X_test)
    y_test_vector = [y[0] for y in y_test.T.tolist()]
    print_pred_vs_obs(predictions, y_test_vector)
    print("precision:", prediction_precision(predictions, y_test.T))
    print("mean error:", prediction_mean_error(predictions, y_test.T))
    print("probas[0]:", probas[0])
    print("predictions[0]:", predictions[0])
    return

if __name__ == '__main__':
    main()
