import numpy as np
import math
import read_file
import sigmoid
import sigmoid_gradient
import cost_function
import matplotlib.pyplot as plt

# constants

HIDDEN_LAYER_NODES = 5
OUTPUT_LAYER_NODES = 1
NUM_ITERATIONS = 10000
COST_THRESHOLD = math.pow(10, -3)
INITIAL_EPSILON = 0.05
LEARNING_RATE = 0.05
GRADIENT_CHECK_EPSILON = 0.001

# get training sets

y, new_features_sets, number_features_sets, number_features = read_file.get_features_sets()
training = math.floor(number_features_sets * 0.7)
training_sets = new_features_sets[:training, :]
y_training_sets = y[:training, :]

# -- NN -> input layer: 8 nodes, hidden layer: 5 nodes, output layer: 1 node
# initialize theta

Theta1 = np.random.rand(HIDDEN_LAYER_NODES, number_features) * \
    (2 * INITIAL_EPSILON) - INITIAL_EPSILON
Theta2 = np.random.rand(
    OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES + 1) * 2 * INITIAL_EPSILON - INITIAL_EPSILON

J = 1
# J_datas = []

for i in range(NUM_ITERATIONS):
    # forward propagation

    J, h, new_a2_sets, z2 = cost_function.cost_function(
        training_sets, Theta1, Theta2, number_features_sets, y_training_sets)
    # J_datas.append(J)
    print(J)
    # if(J < COST_THRESHOLD):
    #     break
    # back propagation

    # delta3 = h - y_training_sets.T
    # Theta2_grad = (delta3@new_a2_sets.T)/number_features_sets
    # Fake_theta2 = Theta2[:, 1:]
    # delta2 = (Fake_theta2.T@delta3)*sigmoid_gradient.sigmoid_gradient(z2)
    # Theta1_grad = (delta2@training_sets)/number_features_sets

    # manually compute derivative

    Theta1_grad = np.zeros((Theta1.shape[0], Theta1.shape[1]))
    Theta2_grad = np.zeros((Theta2.shape[0], Theta2.shape[1]))
    for i1 in range(Theta1.shape[0]):
        for i2 in range(Theta1.shape[1]):
            Theta1_clone = Theta1.copy()
            Theta1_clone[i1][i2] = Theta1_clone[i1][i2] + \
                GRADIENT_CHECK_EPSILON
            Theta1_clone_minus = Theta1.copy()
            Theta1_clone_minus[i1][i2] = Theta1_clone_minus[i1][i2] - \
                GRADIENT_CHECK_EPSILON
            J1_clone, h, new_a2_sets, z2 = cost_function.cost_function(
                training_sets, Theta1_clone, Theta2, number_features_sets, y_training_sets)
            J1_clone_minus, h, new_a2_sets, z2 = cost_function.cost_function(
                training_sets, Theta1_clone_minus, Theta2, number_features_sets, y_training_sets)
            Theta1_grad[i1][i2] = (
                J1_clone - J1_clone_minus)/(2*GRADIENT_CHECK_EPSILON)

    for i3 in range(Theta2.shape[0]):
        for i4 in range(Theta2.shape[1]):
            Theta2_clone = Theta2.copy()
            Theta2_clone[i3][i4] = Theta2_clone[i3][i4] + \
                GRADIENT_CHECK_EPSILON
            Theta2_clone_minus = Theta2.copy()
            Theta2_clone_minus[i3][i4] = Theta2_clone_minus[i3][i4] - \
                GRADIENT_CHECK_EPSILON
            J2_clone, h, new_a2_sets, z2 = cost_function.cost_function(
                training_sets, Theta1, Theta2_clone, number_features_sets, y_training_sets)
            J2_clone_minus, h, new_a2_sets, z2 = cost_function.cost_function(
                training_sets, Theta1, Theta2_clone_minus, number_features_sets, y_training_sets)
            Theta2_grad[i3][i4] = (
                J2_clone - J2_clone_minus)/(2*GRADIENT_CHECK_EPSILON)

    # gradient descent

    Theta1 = Theta1 - LEARNING_RATE*Theta1_grad
    Theta2 = Theta2 - LEARNING_RATE*Theta2_grad

# np.savetxt("Theta1.csv", Theta1, delimiter=",")
# np.savetxt("Theta2.csv", Theta2, delimiter=",")
# plt.plot(J_datas)
# plt.xlabel("Number of iterations")
# plt.ylabel("Cost function")
# plt.show()
