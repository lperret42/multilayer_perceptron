#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
import matplotlib.pyplot as plt
from src import dataframe

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    df = dataframe.read_csv(args.csvfile)
    df.get_numerical_features()
    df.remove_nan()
    df.standardize()
    df_standardized = dataframe.DataFrame(data=df.standardized)
    dfs_by_diagnosis = {diagnosis: df_standardized.get_df_filtered({"diagnosis": diagnosis}) for
            diagnosis in list(set(df.data["diagnosis"]))}
    for feature in df.numerical_features:
        values = df.data[feature]
        to_plot = [df.data[feature] for diagnosis, df in dfs_by_diagnosis.items()]
        plt.hist(to_plot)
        plt.xlabel("Notes")
        plt.ylabel("Frequency")
        plt.legend([diagnosis for diagnosis, _ in dfs_by_diagnosis.items()])
        plt.title(feature)
        plt.show()

if __name__ == '__main__':
    main()
