import numpy as np


def get_features_sets():
    data = np.genfromtxt("data_sets.csv", delimiter=",")
    features_sets = data[:, :data.shape[1] - 1]
    y = data[:, (data.shape[1] - 1): (data.shape[1])]
    features_sets_bias = np.ones((features_sets.shape[0], 1))
    new_features_sets = np.concatenate(
        (features_sets_bias, features_sets), axis=1)
    number_features_sets = new_features_sets.shape[0]
    number_features = new_features_sets.shape[1]

    return y, new_features_sets, number_features_sets, number_features
