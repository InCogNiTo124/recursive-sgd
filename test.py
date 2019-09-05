import numpy as np
import csv
#from layers import Sigmoid
from layers import Tanh as Sigmoid
from losses import MSE
LAYER_LIST = [2, 3, 1]
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
    def __init__(self, in_dim, out_dim, activation=Sigmoid()):
        assert in_dim > 0
        assert out_dim > 0
        assert type(in_dim) == type(0) # INT
        assert type(out_dim) == type(0) # INT
        self.lin = Linear(in_dim, out_dim)
        self.nlin = activation
        return

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
        print("y_true {} \n y_pred {} \n\t\t loss {}".format(y, X, loss.forward(y, X)))
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
    layers = [Layer(i, o) for i, o in zip(LAYER_LIST, LAYER_LIST[1:])]
    loss = MSE()
    dataset[dataset[:, 2] == 0, 2] = -1
    sgd(dataset[:, :2], dataset[:, 2:],
        layers,
        loss,
        batch_size=32,
        epochs=1000)
    #TODO:
    # - refactor layer class
    # - add Tanh and relu activation
    # - make python cli interface

