import argparse
import json
import numpy as np
from src import dataframe
from src.mlp import Mlp
from src.utils import train_test_split, print_pred_vs_obs,\
                      prediction_precision, prediction_mean_error

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    output_label = 'iris'
    output_label = 'diagnosis'
    output_label = 'age'
    args = parse_arguments()
    df = dataframe.DataFrame.read_csv(args.csvfile)
    df.set_numerical_features(to_remove=["age"])
    #df.set_numerical_features(to_remove=["id"])
    df.digitalize()
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    y = df.data[output_label]
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.8)
    unique = np.unique(np.asarray(y_train))
    dim_input, dim_output = X_train.shape[0], np.unique(np.asarray(y_train)).shape[0]
    mlp = Mlp(dim_input, dim_output,
              hidden_layer_sizes=None)
    mlp.fit(X_train, y_train, verbose=True)
    mlp.dump(model_name=output_label+".pkl")
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
