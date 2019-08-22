import numpy as np
import csv
LAYER_LIST = [2, 3, 1]
LEARNING_RATE=1e-3
np.random.seed(0)

class MSE():
    def __init__(self):
        return

    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)


    def backward(self, y_true, y_pred):
        #print("y_true.shape", y_true.shape)
        #print("y_pred.shape", y_pred.shape)
        #print(y_pred)
        return np.mean(y_true - y_pred, axis=1, keepdims=True)

class Linear():
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(out_dim, in_dim)
        self.b = np.random.randn(out_dim, 1)
        return

    def forward(self, X):
        return self.W.dot(X) + self.b

    def backward(self, delta):
        #print(self.W.shape)
        #print(delta.shape)
        return self.W.T.dot(delta)

class Sigmoid():
    def __init__(self):
        return

    def forward(self, X):
        return 1/(1+np.exp(-X))

    def backward(self, delta):
        return delta*(1-delta)

class Layer():
    def __init__(self, in_dim, out_dim):
        self.lin = Linear(in_dim, out_dim)
        self.nlin = Sigmoid()
        return

def sgd(X, y_true, layer_list, loss, batch_size=4, epochs=1):
    #print("X.shape {}\ny_true.shape {}".format(X.shape, y_true.shape))
    for epoch in range(1, epochs+1):
        print("EPOCH: {}".format(epoch))
        for i in range(1):#X.shape[0]//batch_size):
            start = i*batch_size
            end = start + batch_size
            X_batch = X[start:end, :]
            y_batch = y_true[start:end, :]
            sgd_step(X_batch.T, y_batch.T, layer_list, loss)
    return

def sgd_step(X, y, layer_list, loss):
    #print("y", y)
    if layer_list == []:
        #print(loss.forward(X, y))
        print(loss.backward(X, y).shape)
        return loss.backward(X, y)
    else:
        layer = layer_list[0]
        z = layer.lin.forward(X)
        a = layer.nlin.forward(z)
        #print("a.shape", a.shape)
        grad = sgd_step(a, y, layer_list[1:], loss)
        #print("grad.shape", grad.shape)
        # TODO: update
        n_grad = layer.nlin.backward(grad)
        print("n_grad.shape", n_grad.shape)
        dw = X.T * n_grad
        #print(dw.shape)
        #print(layer.lin.W.shape)
        l_grad = layer.lin.backward(n_grad)
        layer.lin.W -= LEARNING_RATE*dw
        layer.lin.b -= LEARNING_RATE*n_grad
        return l_grad

if __name__ == '__main__':
    with open("dataset.csv", "r") as f:
        dataset = np.array(list(csv.reader(f, delimiter=",")), dtype=np.float64)
    #print(dataset.shape)
    #print(dataset[:10, :])
    layers = [Layer(i, o) for i, o in zip(LAYER_LIST, LAYER_LIST[1:])]
    #print(layers)
    loss = MSE()
    sgd(dataset[:, :2], dataset[:, 2:], layers, loss, epochs=15)

