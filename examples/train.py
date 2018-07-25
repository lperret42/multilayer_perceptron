import argparse
from toolbox import dataframe
from toolbox.utils import train_test_split
from neurals_network.mlp import MlpClassifier

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
        help="describe what append in algorithm")
    parser.add_argument('csvfile', help='data.csv')
    parser.add_argument('output_label', help='name of labels column')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    output_label = args.output_label
    df = dataframe.DataFrame.read_csv(args.csvfile)
    df.set_numerical_features(to_remove=[output_label])
    df.digitalize()
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    y = df.data[output_label]
    X_train, y_train, _, _ = train_test_split(X, y, train_ratio=1.0)
    mlp = MlpClassifier()
    mlp.fit(X_train, y_train, verbose=args.verbose)
    model_name = output_label + ".pkl"
    mlp.dump(model_name=model_name, verbose=True)
    return

if __name__ == '__main__':
    main()
