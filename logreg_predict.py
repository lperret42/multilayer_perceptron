#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import csv
import json
import argparse
from src import dslr
from src.math import logistic_function, scalar_product

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='dataset_test.csv')
    parser.add_argument('weights', help='weights.json')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    df = dslr.read_csv(args.csvfile)
    del df.data["Hogwarts House"]
    df.get_numerical_features()
    df.digitalize()
    df.replace_nan()
    weights = json.load(open(args.weights))
    theta_by_house = {house: [weights[house][feature] for feature in df.numerical_features] for
            house, features in weights.items()}
    predictions = []
    probas = {}
    for i in range(len(df.data["Index"])):
        x = [df.data[feature][i] for feature in df.numerical_features]
        for house, _ in weights.items():
            cte = weights[house]["cte"]
            theta = theta_by_house[house]
            probas[house] = logistic_function(cte + scalar_product(theta, x))
        predict_house = "Gryffindor"
        proba_max = probas["Gryffindor"]
        for house, proba in probas.items():
            if proba > proba_max:
                proba_max = proba
                predict_house = house

        predictions.append([int(df.data["Index"][i]), predict_house])

    with open('houses.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Index', 'Hogwarts House'],
                                                        lineterminator='\n')
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({'Index': prediction[0], 'Hogwarts House':prediction[1]})
        csvfile.close() 

if __name__ == '__main__':
    main()
