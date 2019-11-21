import numpy as np

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

    def backward(self, X, grad):
        a = self.forward(X)
        return a * (1-a) * grad

class Tanh(BaseTransform):
    def __init__(self):
        return

    def forward(self, X):
        return np.tanh(X)

    def backward(self, X, grad):
        a = self.forward(X)
        return (1-a) * (1+a) * grad

class ReLU(BaseTransform):
    def __init__(self):
        super().__init__()
        return

    def forward(self, X):
        return X * (X >= 0)

    def backward(self, X, grad):
        return (X >= 0) * grad

class Linear(BaseTransform):
    def __init__(self, in_dim, out_dim):
        super().__init__(True)
        self.W = np.random.randn(out_dim, in_dim)*0.5
        return

    def forward(self, X):
        return self.W.dot(X)

    def backward(self, X, grad):
        return self.W.T @ grad

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

    def backward(self, X, grad):
        return grad

    def update(self, X, delta, LR):
        self.b -= LR * np.mean(delta, axis=1, keepdims=True)
        return

    def get_weights(self):
        return self.b

