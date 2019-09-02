import numpy as np

class Sigmoid():
    def __init__(self):
        return

    def forward(self, X):
        return 1/(1+np.exp(-X))

    def backward(self, delta):
        return delta*(1-delta)

class Tanh():
    def __init__(self):
        return

    def forward(self, X):
        return np.tanh(X)

    def backward(self, delta):
        return (1-delta) * (1+delta)

