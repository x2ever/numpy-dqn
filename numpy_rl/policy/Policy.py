from abc import *

import numpy as np

class Policy(metaclass = ABCMeta):
    def __init__(self, input_n, output_n, hidden_n=64, hidden_layer_n=2):
        pass

    @abstractmethod
    def train(self, x, y, learning_rate=0.001):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass

    @classmethod
    def ReLU(cls, x):
        return (np.maximum(0, x))

    @classmethod
    def d_ReLU(cls, x):
        return (np.heaviside(x, 1.0))

    @classmethod
    def Sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def d_Sigmoid(cls, x):
        return (1 - cls.Sigmoid(x)) * cls.Sigmoid(x)
