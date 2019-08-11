import numpy as np
import csv
LAYER_LIST = [2, 3, 1]
np.random.seed(0)

def mse(y, y_true):
    return np.mean((y - y_true) ** 2) / 2

def cross_entropy(y, y_true):
    return np.mean(y_true*np.log(y) + (1-y_true)*np.log(1-y))


def sgd(X, y_true, layer_list, batch_size=1, epochs=1):
    for i in range(1):#X.shape[0]/batch_size):
        start = i*batch_size
        end = start + batch_size
        X_batch = X[start:end, :]
        y_batch = X[start:end, :]
        sgd_step(X_batch.T, y_batch.T, layer_list)
    return

def sgd_step(X, y, layer_list):
    if layer_list == []:
        return np.array([1])
    else:
        layer = layer_list[0]
        output = layer.forward(X)
        print(output)
        grad = sgd_step(output, y, layer_list[1:])
    return

if __name__ == '__main__':
    with open("dataset.csv", "r") as f:
        dataset = np.array(list(csv.reader(f, delimiter=",")), dtype=np.float64)
    print(dataset.shape)
    print(dataset[:10, :])
    layers = [Linear(i, o, sigm) for i, o in zip(LAYER_LIST, LAYER_LIST[1:])]
    print(layers)
    sgd(dataset[:, :2], dataset[:, 2], layers)

