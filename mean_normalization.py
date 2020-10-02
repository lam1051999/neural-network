import numpy as np


def mean_normalization(sets):
    m = sets.shape[0]
    column_mean = (sets.sum(axis=0))/m
    full_column_mean = np.array([column_mean]*m)
    column_range = np.amax(sets, axis=0) - np.amin(sets, axis=0)
    full_column_range = np.array([column_range]*m)
    new_sets = (sets - full_column_mean)/full_column_range
    return new_sets
