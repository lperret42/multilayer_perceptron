import argparse
import numpy as np
#
from toolbox import dataframe
from toolbox.utils import train_test_split, print_pred_vs_obs,\
                          pred_accuracy, pred_mean_error
from neurals_network.mlp import MlpClassifier
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='data.csv')
    parser.add_argument('model', help='trained model')
    parser.add_argument('output_label', help='name of labels column')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    output_label = args.output_label
    mlp = MlpClassifier.load(args.model)
    df = dataframe.DataFrame.read_csv(args.csvfile)
    df.set_numerical_features(to_remove=[output_label])
    df.digitalize()
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    y = df.data[output_label]
    _, _, X_test, y_test = train_test_split(X, y, train_ratio=0.)
    pred = mlp.predict(X_test)
    y_test = [y[0] for y in y_test.T.tolist()]
    print_pred_vs_obs(pred, y_test, only_false=False)
    print("\naccuracy: {}%".format(100*round(pred_accuracy(pred, y_test), 5)))
    mean_error = pred_mean_error(pred, y_test)
    if mean_error is not None:
        print("\nmean error: {}".format(round(mean_error, 2)))
    return

if __name__ == '__main__':
    main()
