import numpy as np
import math
import read_file
import sigmoid
import sigmoid_gradient
import cost_function
import matplotlib.pyplot as plt
import mean_normalization

# constants

HIDDEN_LAYER_NODES = 5
OUTPUT_LAYER_NODES = 1
NUM_ITERATIONS = 1000
COST_THRESHOLD = math.pow(10, -3)
INITIAL_EPSILON = 0.05
LEARNING_RATE = 0.02
LEARNING_RATES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
GRADIENT_CHECK_EPSILON = 0.001

# get training sets

y, new_features_sets, number_features_sets, number_features = read_file.get_features_sets()
training = math.floor(number_features_sets * 0.7)
training_sets = new_features_sets[:training, :]
y_training_sets = y[:training, :]

training_sets_mean = mean_normalization.mean_normalization(
    training_sets[:, 1:])

new_training_sets = np.concatenate(
    (np.ones((training_sets.shape[0], 1)), training_sets_mean), axis=1)


# -- NN -> input layer: 8 nodes, hidden layer: 5 nodes, output layer: 1 node
# initialize theta

Theta1 = np.random.rand(HIDDEN_LAYER_NODES, number_features) * \
    (2 * INITIAL_EPSILON) - INITIAL_EPSILON
Theta2 = np.random.rand(
    OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES + 1) * 2 * INITIAL_EPSILON - INITIAL_EPSILON

# J_datas = []

# for j in range(len(LEARNING_RATES)):
# J_data = []
for i in range(NUM_ITERATIONS):
    # forward propagation

    J, h, new_a2_sets, z2 = cost_function.cost_function(
        training_sets, Theta1, Theta2, number_features_sets, y_training_sets)
    # J_data.append(J)
    print(J)
    if(J < COST_THRESHOLD):
        break
    # back propagation

    delta3 = h - y_training_sets.T
    Theta2_grad = (delta3@new_a2_sets.T)/number_features_sets
    Fake_theta2 = Theta2[:, 1:]
    delta2 = (Fake_theta2.T@delta3)*sigmoid_gradient.sigmoid_gradient(z2)
    Theta1_grad = (delta2@training_sets)/number_features_sets

    # manually compute derivative

    # grad_theta1_clone = np.zeros((Theta1.shape[0], Theta1.shape[1]))
    # for i1 in range(Theta1.shape[0]):
    #     for i2 in range(Theta1.shape[1]):
    #         Theta1_clone = Theta1.copy()
    #         Theta1_clone[i1][i2] = Theta1_clone[i1][i2] + \
    #             GRADIENT_CHECK_EPSILON
    #         Theta1_clone_minus = Theta1.copy()
    #         Theta1_clone_minus[i1][i2] = Theta1_clone_minus[i1][i2] - \
    #             GRADIENT_CHECK_EPSILON
    #         J1_clone, h, new_a2_sets, z2 = cost_function.cost_function(
    #             training_sets, Theta1_clone, Theta2, number_features_sets, y_training_sets)
    #         J1_clone_minus, h, new_a2_sets, z2 = cost_function.cost_function(
    #             training_sets, Theta1_clone_minus, Theta2, number_features_sets, y_training_sets)
    #         grad_theta1_clone[i1][i2] = (
    #             J1_clone - J1_clone_minus)/(2*GRADIENT_CHECK_EPSILON)

    # gradient descent

    Theta1 = Theta1 - LEARNING_RATE*Theta1_grad
    Theta2 = Theta2 - LEARNING_RATE*Theta2_grad
    # J_datas.append(J_data)
np.savetxt("Theta1.csv", Theta1, delimiter=",")
np.savetxt("Theta2.csv", Theta2, delimiter=",")
# for i in range(len(J_datas)):
#     plt.plot(J_datas[i])
#     plt.xlabel("Number of iterations")
#     plt.ylabel(f"Cost function for learning rate = {LEARNING_RATES[i]}")
#     plt.show()
