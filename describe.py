#!/Users/lperret/.brew/Cellar/python/3.6.5/bin/python3.6

import argparse
from src import dataframe

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='data.csv')
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    df = dataframe.read_csv(args.csvfile)
    df.describe()

if __name__ == '__main__':
    main()
