import numpy as np
from rop import read_only_properties

@read_only_properties('trainable')
class BaseTransform():
    def __init__(self, trainable=False, next_one=None):
        self.trainable = trainable
        self.next = next_one
        return

    def forward(self, X):
        pass

    def backward(self, delta):
        pass

class Sigmoid(BaseTransform):
    def __init__(self):
        super().__init__()
        return

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward(self, X):
        a = self.forward(X)
        return np.diagflat(a * (1 - a))

class Tanh():
    def __init__(self):
        return

    def forward(self, X):
        return np.tanh(X)

    def backward(self, delta):
        return (1-delta) * (1+delta)

class ReLU():
    def __init__(self):
        return

    def forward(self, X):
        return X * (X >= 0)

    def backward(self, delta):
        return (delta >= 0).astype(np.float64)

class Linear(BaseTransform):
    def __init__(self, in_dim, out_dim):
        super().__init__(True)
        self.W = np.random.randn(out_dim, in_dim)*0.1
        return

    def forward(self, X):
        return self.W.dot(X)

    def backward(self, X):
        return self.W.T

    def update(self, X, delta, LR):
        dw = delta @ X.T
        self.W -= LR * dw
        return

    def get_weights(self):
        return self.W

class AddBias(BaseTransform):
    def __init__(self, in_dim):
        super().__init__(True)
        self.b = np.random.randn(in_dim, 1) * 0.1
        return

    def forward(self, X):
        return X + self.b

    def backward(self, X):
        #return delta
        return np.eye(X.shape[1], dtype=np.float64)

    def update(self, X, delta, LR):
        self.b -= LR * np.mean(delta, axis=1, keepdims=True)
        return

    def get_weights(self):
        return self.b
