from __future__ import print_function
import random
from collections import OrderedDict
import numpy as np
from toolbox.utils import get_data, keep_only_float, quicksort, is_float,\
                   is_list_num, sum_with_empty, mean_with_empty

class DataFrame(object):
    def __init__(self, data=OrderedDict()):
        self.data = data
        self.description = {}
        self.standardized= OrderedDict()
        self.stand_coefs = {}
        self.numerical_features = []

    @classmethod
    def read_csv(cls, csvfile):
        df = DataFrame()
        df.data = get_data(csvfile)
        return df

    def set_numerical_features(self, to_remove=[]):
        self.numerical_features = []
        for feature, values in self.data.items():
            if feature not in to_remove and is_list_num(values):
                self.numerical_features.append(feature)

    def get_description(self):
        description = {}
        for feature, values in self.data.items():
            if feature not in self.numerical_features:
                continue
            only_float = keep_only_float(values)
            if len(only_float) > 0:
                float_sorted = quicksort(only_float)
                description.setdefault("Field", []).append(feature)
                description.setdefault("Count", []).append(float(len(float_sorted)))
                description.setdefault("Mean", []).append(np.mean(float_sorted))
                description.setdefault("Std", []).append(np.std(float_sorted))
                description.setdefault("Min", []).append(min(float_sorted))
                description.setdefault("25%", []).append(np.percentile(
                                                            float_sorted, 25))
                description.setdefault("50%", []).append(np.percentile(
                                                            float_sorted, 50))
                description.setdefault("75%", []).append(np.percentile(
                                                            float_sorted, 75))
                description.setdefault("Max", []).append(max(float_sorted))

        return description

    def describe(self):
        self.set_numerical_features()
        if self.description == {}:
            self.description = self.get_description()
        nb = len(self.description["Count"])
        order = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        print("{:<10}".format(""), end="")
        for n in range(nb):
            print("{:>17}".format(self.description["Field"][n][:15]), end="")
        print('\n', end="")
        for elem in order:
            features = self.description[elem]
            print("{:<10}".format(elem), end="")
            for n in range(nb):
                print("{:17.6f}".format(round(features[n], 6)), end="")
            print('\n', end="")

    def filter(self, feature_value_to_filter={}, to_keep=True):
        index_to_del = []
        for feature, value in feature_value_to_filter.items():
            values = self.data[feature]
            index_to_del += [i for i in range(len(values)) if values[i] != value] if\
                to_keep == True else [i for i in range(len(values))
                    if values[i] == value]
        index_to_del = list(set(index_to_del))
        filtered_data = {feature: [values[i] for i in range(len(values)) if
            i not in index_to_del] for (feature, values) in self.data.items()}
        return filtered_data

    def get_df_filtered(self, feature_value_to_filter={}):
        df = DataFrame(data=self.filter(feature_value_to_filter=feature_value_to_filter))
        return df

    def remove_nan(self, exception=""):
        self.data = self.filter(feature_value_to_filter={feature:"" for (feature, _) in
            self.data.items() if feature != exception}, to_keep=False)

    def replace_nan(self):
        for feature, values in self.data.items():
            if not feature in self.numerical_features:
                continue
            m = mean_with_empty(values)
            self.data[feature] = [value if value != "" else m for value in values]

    def digitalize(self):
        for feature, values in self.data.items():
            if is_float(values[0]):
                self.data[feature] = [float(v) if v != "" else v for v in values]

    def standardize(self):
        for feature, values in self.data.items():
            if feature not in self.numerical_features:
                self.standardized[feature] = values
                continue
            only_float = keep_only_float(values)
            if len(only_float) > 0:
                mu = np.mean(only_float)
                sigma = np.sqrt(sum([(x - mu)**2 for x in only_float]))
                self.standardized[feature] = [(x - mu) / sigma for x in only_float]
                self.stand_coefs[feature] = {"mu": mu, "sigma":sigma}
            else:
                self.standardized[feature] = values
