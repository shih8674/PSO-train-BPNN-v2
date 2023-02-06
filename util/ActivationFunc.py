import numpy as np


class ActFunc:
    def relu(x):
        if x < 0: return 0
        else: return x

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))