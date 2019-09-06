import numpy as np
import csv
from layers import Sigmoid, ReLU, Tanh
from losses import MSE, CE
ACTIVATION_DICT = {"relu": ReLU(), "sigmoid": Sigmoid(), "tanh": Tanh()}

LAYER_LIST = [2, 8, 14, 6, 1]
ACTIVATION_LIST = ["tanh", "sigmoid", "tanh", "sigmoid"]
LEARNING_RATE = 0.1
np.random.seed(0)


class Linear():
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(out_dim, in_dim)*0.5
        self.b = np.random.randn(out_dim, 1)*0.5
        return

    def forward(self, X):
        return self.W.dot(X) + self.b

    def backward(self, delta):
        return self.W.T.dot(delta)

class Layer():
    def __init__(self, in_dim, out_dim, activation=None):
        assert in_dim > 0
        assert out_dim > 0
        assert type(in_dim) == type(0) # INT
        assert type(out_dim) == type(0) # INT
        self.lin = Linear(in_dim, out_dim)
        self.nlin = activation if activation is not None else Sigmoid()
        return

    def forward(self, X):
        return self.nlin.forward(self.lin.forward(X))


def sgd(X, y_true, layer_list, loss, batch_size=4, epochs=1):
    for epoch in range(1, epochs+1):
        #print("EPOCH: {}".format(epoch))
        for i in range(X.shape[0]//batch_size):
            start = i*batch_size
            end = start + batch_size
            X_batch = X[start:end, :]
            y_batch = y_true[start:end, :]
            sgd_step(X_batch.T, y_batch.T, layer_list, loss)
    return

def sgd_step(X, y, layer_list, loss):
    if layer_list == []:
        print("y_true {} \n y_pred {} \n\t\t loss {}".format(y, X, loss.forward(X, y)))
        return loss.backward(X, y)
    else:
        layer = layer_list[0]
        z = layer.lin.forward(X)
        a = layer.nlin.forward(z)
        grad = sgd_step(a, y, layer_list[1:], loss)
        n_grad = grad * layer.nlin.backward(a)
        dw = n_grad.dot(X.T)
        assert dw.shape == layer.lin.W.shape
        l_grad = layer.lin.W.T @ n_grad
        layer.lin.W -= LEARNING_RATE*dw
        return l_grad

if __name__ == '__main__':
    with open("dataset.csv", "r") as f:
        dataset = np.array(list(csv.reader(f, delimiter=",")), dtype=np.float64)
    layers = [Layer(i, o, ACTIVATION_DICT[a]) for i, o, a in zip(LAYER_LIST, LAYER_LIST[1:], ACTIVATION_LIST)]
    #loss = MSE()
    loss = CE()
    #dataset[dataset[:, 2] == 0, 2] = -1
    #dataset = dataset[:8, :]
    sgd(dataset[:, :2], dataset[:, 2:],
        layers,
        loss,
        batch_size=8,
        epochs=200)
    # TESTING
    x = np.arange(0, 1, 1/100)
    y = np.arange(0, 1, 1/100)
    a = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    y_test = a.copy().T
    for layer in layers:
       y_test = layer.forward(y_test)
    THR = 0.5 
    y_test[y_test >= THR] = 1
    y_test[y_test < THR] = 0

    import matplotlib.pyplot as plt
    plt.scatter(a[:, 0], a[:, 1], c=y_test.flatten().astype(int), s=1)
    plt.show()

    #TODO:
    # - refactor layer class
    # - add Tanh and relu activation
    # - make python cli interface

