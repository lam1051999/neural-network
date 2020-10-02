import numpy as np
import sigmoid


def sigmoid_gradient(z):
    g1 = sigmoid.sigmoid(z)
    return g1 * (1 - g1)
