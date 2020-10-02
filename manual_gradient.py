import numpy as np
import cost_function


def manual_gradient(Theta1, Theta2, GRADIENT_CHECK_EPSILON, training_sets, number_features_sets, y_training_sets):
    grad_theta1_clone = np.zeros((Theta1.shape[0], Theta1.shape[1]))
    for i1 in range(Theta1.shape[0]):
        for i2 in range(Theta1.shape[1]):
            Theta1_clone = Theta1
            Theta1_clone[i1][i2] = Theta1_clone[i1][i2] + \
                GRADIENT_CHECK_EPSILON
            Theta1_clone_minus = Theta1
            Theta1_clone_minus[i1][i2] = Theta1_clone[i1][i2] - \
                GRADIENT_CHECK_EPSILON
            J1_clone, h, new_a2_sets, z2 = cost_function.cost_function(
                training_sets, Theta1_clone, Theta2, number_features_sets, y_training_sets)
            J1_clone_minus, h, new_a2_sets, z2 = cost_function.cost_function(
                training_sets, Theta1_clone_minus, Theta2, number_features_sets, y_training_sets)
            grad_theta1_clone[i1][i2] = (
                J1_clone - J1_clone_minus)/(2*GRADIENT_CHECK_EPSILON)
    return grad_theta1_clone
