import numpy as np
import csv
#from losses import CE
#from layers import Tanh as Sigmoid
from layers import Sigmoid
from losses import MSE
LAYER_LIST = [2, 3, 1]
#LEARNING_RATE=0.5 #1e-3
LEARNING_RATE = 1e-1
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
    #print("X.shape {}\ny_true.shape {}".format(X.shape, y_true.shape))
    for epoch in range(1, epochs+1):
        #print("EPOCH: {}".format(epoch))
        for i in range(X.shape[0]//batch_size):
            start = i*batch_size
            end = start + batch_size
            X_batch = X[start:end, :]
            y_batch = y_true[start:end, :]
            sgd_step(X_batch.T, y_batch.T, layer_list, loss)
        #print("RESIDUAL i =", i)
    return

def sgd_step(X, y, layer_list, loss):
    if layer_list == []:
        print("y_true {} \n y_pred {} \n\t\t loss {}".format(y, X, loss.forward(y, X)))
        #print(loss.forward(X, y))
        #print(X)
        #print()
        return loss.backward(X, y)
    else:
        layer = layer_list[0]
        z = layer.lin.forward(X)
        #print("z", z)
        a = layer.nlin.forward(z)
        #print("a", a)
#       #print("W.shape {} \t X.shape {} \t a.shape {}".format(layer.lin.W.shape, X.shape, a.shape))
        grad = sgd_step(a, y, layer_list[1:], loss)
        #print("g", grad)
        # TODO: update
        #print("s", layer.nlin.backward(a))
        #print("X", X)
        n_grad = grad * layer.nlin.backward(a)
        dw = n_grad.dot(X.T)
        #print(grad.shape)
        #print(n_grad.shape)
        #print(layer.nlin.backward(a).shape)
        #print(X.shape)
        #raise ValueError
        #dw = np.mean(X.T * n_grad, axis=1, keepdims=Tru
        #print("dw", dw)
        assert dw.shape == layer.lin.W.shape
        l_grad = layer.lin.W.T @ n_grad
        layer.lin.W -= LEARNING_RATE*dw
        #print("\t NEW W\n", layer.lin.W)
        #layer.lin.b -= LEARNING_RATE*np.mean(n_grad, axis=1, keepdims=True)
        return l_grad

if __name__ == '__main__':
    with open("dataset.csv", "r") as f:
        dataset = np.array(list(csv.reader(f, delimiter=",")), dtype=np.float64)
    #print(dataset.shape)
    #print(dataset[:10, :])
    layers = [Layer(i, o) for i, o in zip(LAYER_LIST, LAYER_LIST[1:])]
    #layers[0].lin.W = np.array([[0.15, 0.2], [0.25, 0.30]])
    #layers[0].lin.b = np.array([[0.35], [0.35]])
    #layers[1].lin.W = np.array([[0.40, 0.45], [0.50, 0.55]])
    #layers[1].lin.b = np.array([[0.60], [0.60]])
    #loss = CE()
    loss = MSE()
    # tanh trick
#    dataset[dataset[:, 2] == 0, 2] = -1
#    print(dataset[:10, :])
    #dataset = dataset[:8, :]
    sgd(dataset[:, :2], dataset[:, 2:],
    #sgd(np.array([[0.05, 0.10]]),
    #    np.array([[0.01, 0.99]]),
        layers,
        loss,
        batch_size=8,
        epochs=1000)
    #TODO: add another fake example
