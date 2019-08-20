import numpy as np

class Loss():
    def __init__(self):
        return

    def forward(self, y_true, y_pred):
        pass

    def backward(self, y_true, y_pred):
        pass

class MeanSquaredError():
    def __init__(self):
        super().__init__(self)
        return

    def forward(self, y_true, y_pred):
        return np.sum((y_true - y_pred)**2) / y_true.shape[1]

    def backward(self, y_true, y_pred):
        return
