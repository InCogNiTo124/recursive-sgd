import numpy as np

class Linear():
    def __init__(self, in_dimension, out_dimension):
        self.W = np.random.randn(out_dimension, in_dimension)
        return

    def forward(self, X):
        return self.W.dot(X)

    def backward(self):
        pass

    def __repr__(self):
        return "Linear(W=\n{}\nb=\n{})".format(repr(self.W), repr(self.b))
class Tanh():

def tanh(x):
    return np.tanh(x)

def sigm(x):
    return 1/(1+np.exp(-x))

def relu(x):
    # pretty f***ing smart :D
    return x * (x > 0)

