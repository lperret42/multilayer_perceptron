#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
from sklearn.metrics import accuracy_score
from src.logistic_regressor import LogisticRegressor
from src import dataframe
from src.math import logistic_function, scalar_product

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='dataset_train.csv')
    args = parser.parse_args()

    return args

def transform_label(Y, label):
    return [1 if y == label else 0 for y in Y]

def get_X(data, features):
    X = []
    for i in range(len(features[0])):
        X.append([data[feature][i] for feature in features])
    return X

def main():
    args = parse_arguments()
    df = dataframe.read_csv(args.csvfile)
    df.get_numerical_features()
    df.digitalize()
    df.replace_nan()
    df_train, df_test = df.train_test_split()
    df_train.get_numerical_features()
    df_test.get_numerical_features()
    df_train.standardize()
    df_train.digitalize()
    df_test.digitalize()
    print("len train:", len(df_train.data["diagnosis"]))
    print("len test:", len(df_test.data["diagnosis"]))
    data = df_train.standardized
    to_train = {
            "M": df.numerical_features,
            "B": df.numerical_features,
    }
    to_save = {}
    for diagnosis, features in to_train.items():
        to_save[diagnosis] = {}
        Y = transform_label(data["diagnosis"], diagnosis)
        X = get_X(data, features)
        logistic_regressor = LogisticRegressor(X, Y)
        if args.verbose:
            print("Training one classifier on class", diagnosis)
        logistic_regressor.train(print_cost=args.verbose, max_iter=2000)
        theta = [logistic_regressor.theta[i] / df_train.stand_coefs[feature]["sigma"] for
                i, feature in enumerate(features)]
        cte = -sum([df_train.stand_coefs[feature]["mu"] * logistic_regressor.theta[i] /
            df_train.stand_coefs[feature]["sigma"] for i, feature in enumerate(features)])
        to_save[diagnosis]["cte"] = cte
        for i, feature in enumerate(features):
            to_save[diagnosis][feature] = theta[i]
        for feature in df.numerical_features:
            to_save[diagnosis].setdefault(feature, 0)
    nb_error = 0
    Y_true = df_test.data["diagnosis"]
    Y_pred = []
    predictions = {}
    for i, real_diagnosis in enumerate(df_test.data["diagnosis"]):
        x = [df_test.data[feature][i] for feature in df.numerical_features]
        for diagnosis in list(set(df_test.data["diagnosis"])):
            cte = to_save[diagnosis]["cte"]
            theta = [to_save[diagnosis][feature] for feature in df.numerical_features]
            predictions[diagnosis] = logistic_function(cte + scalar_product(theta, x))

        predict_diagnosis = "M"
        proba_max = predictions["M"]
        for diagnosis, proba in predictions.items():
            if proba > proba_max:
                proba_max = proba
                predict_diagnosis = diagnosis
        Y_pred.append(predict_diagnosis)
        if predict_diagnosis != real_diagnosis:
            print(predictions, "real:", real_diagnosis, "  predict:", predict_diagnosis)
            nb_error += 1

    print("precision:", 1 - nb_error  / len(df_test.data["diagnosis"]))
    print(accuracy_score(Y_true, Y_pred))

if __name__ == '__main__':
    main()
