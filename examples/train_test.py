import argparse
import numpy as np
from toolbox import dataframe
from toolbox.utils import train_test_split, print_pred_vs_obs, pred_accuracy
from neurals_network.mlp import MlpClassifier

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    output_label = 'iris'
    args = parse_arguments()
    df = dataframe.DataFrame.read_csv(args.csvfile)
    df.set_numerical_features()
    df.digitalize()
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    y = df.data[output_label]
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.8)
    unique = np.unique(np.asarray(y_train))
    dim_input, dim_output = X_train.shape[0], np.unique(np.asarray(y_train)).shape[0]
    mlp = MlpClassifier(dim_input, dim_output,
              hidden_layer_sizes=None)
    mlp.fit(X_train, y_train, verbose=args.verbose)
    model_name = output_label + ".pkl"
    mlp.dump(model_name=model_name)
    mlp = MlpClassifier.load("models/" + model_name)
    print("\nTest:")
    predictions = mlp.predict(X_test)
    y_test = [y[0] for y in y_test.T.tolist()]
    print_pred_vs_obs(predictions, y_test)
    print("\nprecision:", pred_accuracy(predictions, y_test))
    return

if __name__ == '__main__':
    main()
