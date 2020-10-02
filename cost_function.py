import numpy as np
import sigmoid


def cost_function(training_sets, Theta1, Theta2, number_features_sets, y_training_sets):
    # z2 = training_sets @ Theta1.T
    # a2 = sigmoid.sigmoid(z2)
    # hidden_sets_bias = np.ones((a2.shape[0], 1))
    # new_a2_sets = np.concatenate((hidden_sets_bias, a2), axis=1)
    # z3 = new_a2_sets @ Theta2.T
    # h = sigmoid.sigmoid(z3)
    z2 = Theta1 @ training_sets.T
    a2 = sigmoid.sigmoid(z2)
    hidden_sets_bias = np.ones((1, a2.shape[1]))
    new_a2_sets = np.concatenate((hidden_sets_bias, a2), axis=0)
    z3 = Theta2 @ new_a2_sets
    h = sigmoid.sigmoid(z3)

    J = (-1/number_features_sets)*np.sum(y_training_sets *
                                         np.log(h.T) + (1-y_training_sets)*np.log(1-h.T))
    return J, h, new_a2_sets, z2
