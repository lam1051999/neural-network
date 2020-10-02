import numpy as np
import math
import read_file
import cost_function
import mean_normalization

# get test sets

y, new_features_sets, number_features_sets, number_features = read_file.get_features_sets()

training = math.floor(number_features_sets * 0.7)
test = number_features_sets - training
test_sets = new_features_sets[training:, :]
y_test_sets = y[training:, :]

test_sets_mean = mean_normalization.mean_normalization(test_sets[:, 1:])

new_test_sets = np.concatenate(
    (np.ones((test_sets.shape[0], 1)), test_sets_mean), axis=1)

# Theta

Theta1 = np.genfromtxt("Theta1.csv", delimiter=",")
Theta2 = np.genfromtxt("Theta2.csv", delimiter=",")

theta2 = np.array([Theta2])

J, h, new_a2_sets, z2 = cost_function.cost_function(
    new_test_sets, Theta1, theta2, number_features_sets, y_test_sets)

result = h.T
result[result >= 0.5] = 1
result[result < 0.5] = 0
compare = result == y_test_sets
num_right_predictions = compare[compare == True].size
accuracy = (num_right_predictions/y_test_sets.size)*100

print("Test sets accuracy: %.2f" % accuracy + "%")
