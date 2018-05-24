from src.utils import get_data, keep_only_float, quicksort, is_float, is_list_num
from src.math import min, max, mean, std, quartile_n, sqrt, sum_with_empty, mean_with_empty
import random

def read_csv(csvfile):
    df = DataFrame()
    df.data = get_data(csvfile)
    return df

class DataFrame(object):
    def __init__(self, data={}):
        self.data = data
        self.description = {}
        self.standardized= {}
        self.stand_coefs = {}
        self.numerical_features = []

    def get_numerical_features(self, remove=["id"]):
        self.numerical_features = []
        for feature, values in self.data.items():
            if feature not in remove and is_list_num(values):
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
                description.setdefault("Mean", []).append(mean(float_sorted))
                description.setdefault("Std", []).append(std(float_sorted))
                description.setdefault("Min", []).append(min(float_sorted))
                description.setdefault("25%", []).append(quartile_n(float_sorted, 1))
                description.setdefault("50%", []).append(quartile_n(float_sorted, 2))
                description.setdefault("75%", []).append(quartile_n(float_sorted, 3))
                description.setdefault("Max", []).append(max(float_sorted))

        return description

    def describe(self):
        self.get_numerical_features()
        if self.description == {}:
            self.description = self.get_description()
        nb = len(self.description["Count"])
        order = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        print("{:<10}".format(""), end="")
        [print("{:>17}".format(self.description["Field"][n][:15]), end="") for n in range(nb)]
        print("")
        for elem in order:
            features = self.description[elem]
            print("{:<10}".format(elem), end="")
            [print("{:17.6f}".format(round(features[n], 6)), end="") for n in range(nb)]
            print("")

    def filter(self, feature_value_to_filter={}, to_keep=True):
        index_to_del = []
        for feature, value in feature_value_to_filter.items():
            values = self.data[feature]
            index_to_del += [i for i in range(len(values)) if values[i] != value] if\
                to_keep == True else [i for i in range(len(values)) if values[i] == value]
        index_to_del = list(set(index_to_del))
        filtered_data = {feature: [values[i] for i in range(len(values)) if i not in index_to_del] for\
                (feature, values) in self.data.items()}
        return filtered_data

    def get_df_filtered(self, feature_value_to_filter={}):
        df = DataFrame(data=self.filter(feature_value_to_filter=feature_value_to_filter))
        return df

    def remove_nan(self, exception="diagnosis"):
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
                self.data[feature] = [float(value) if value != "" else value for value in values]

    def standardize(self):
        for feature, values in self.data.items():
            if feature not in self.numerical_features:
                self.standardized[feature] = values
                continue
            only_float = keep_only_float(values)
            if len(only_float) > 0:
                mu = mean(only_float)
                sigma = sqrt(sum([(x - mu)**2 for x in only_float]))
                self.standardized[feature] = [(x - mu) / sigma for x in only_float]
                self.stand_coefs[feature] = {"mu": mu, "sigma":sigma}
            else:
                self.standardized[feature] = values

    def train_test_split(self, train_ratio=0.8):
        nb_rows = len(self.data["id"])
        index = list(range(nb_rows))
        random.shuffle(index)
        index_train = index[:int(train_ratio * nb_rows)]
        index_test = index[int(train_ratio * nb_rows):]
        df_train, df_test = DataFrame(), DataFrame()
        df_train.data = {feature: [values[i] for i in range(len(values)) if i in index_train]
                                    for feature, values in self.data.items()}
        df_test.data = {feature: [values[i] for i in range(len(values)) if i in index_test]
                                    for feature, values in self.data.items()}

        return df_train, df_test
