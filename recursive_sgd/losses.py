import numpy as np

class Loss():
    def __init__(self):
        return

    def forward(self, y_pred, y_true):
        pass

    def backward(self, y_pred, y_true):
        pass

class MSE(Loss):
    def __init__(self):
        super().__init__()
        return

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean((y_true - y_pred)**2)

    def backward(self, y_pred, y_true):
        return y_pred - y_true

class CE(Loss):
    def __init__(self):
        super().__init__()
        return

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / (y_pred * (1- y_pred))
