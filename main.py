import numpy as np
import csv
from sgd.losses import MSE, CE
from sgd.layers import Linear, AddBias, Sigmoid, ReLU
from sgd.sgd import sgd
import logging
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
LEARNING_RATE = 0.05
np.random.seed(0)

class Model():
    def __init__(self, transform=None):
        self.next = transform
        return

    def add_next(self, transform):
        x = self
        while x.next is not None:
            x = x.next
        x.next = transform
        return

if __name__ == '__main__':
    with open("dataset.csv", "r") as f:
        dataset = np.array(list(csv.reader(f, delimiter=",")), dtype=np.float64)

    #layers = [Layer(i, o, ACTIVATION_DICT[a]) for i, o, a in zip(LAYER_LIST, LAYER_LIST[1:], ACTIVATION_LIST)]
    #loss = MSE()
    loss = CE()
    #dataset[dataset[:, 2] == 0, 2] = -1
    #dataset = dataset[1:3, :]
    #dataset = np.array([[0.1, 0.2, 0.5], [0.3, 0.4, 0.6]])
    model = Model()
    """
    linear = Linear(2, 2)
    linear.W = np.array([[0.15, 0.20], [0.25, 0.30]])
    bias = AddBias(2)
    bias.b = np.array([[0.35], [0.35]])
    model.add_next(linear)
    model.add_next(bias)
    model.add_next(Sigmoid())
    linear = Linear(2, 2)
    linear.W = np.array([[0.40, 0.45], [0.50, 0.55]])
    bias = AddBias(2)
    bias.b = np.array([[0.60], [0.60]])
    model.add_next(linear)
    model.add_next(bias)
    model.add_next(Sigmoid())
    #"""
    model.add_next(Linear(2, 15));model.add_next(AddBias(15));model.add_next(ReLU())
    model.add_next(Linear(15, 15));model.add_next(AddBias(15));model.add_next(Sigmoid())
    model.add_next(Linear(15, 15));model.add_next(AddBias(15));model.add_next(ReLU())
    #model.add_next(Linear(15, 15));model.add_next(AddBias(15));model.add_next(ReLU())
    #model.add_next(Linear(6, 3));model.add_next(AddBias(3));model.add_next(Sigmoid())
    #model.add_next(Linear(15, 7));model.add_next(AddBias(7));model.add_next(Sigmoid())
    #model.add_next(Linear(7, 1));model.add_next(AddBias(1));model.add_next(Sigmoid())
    #model.add_next(Linear(3, 1));model.add_next(AddBias(1));model.add_next(Sigmoid())
    model.add_next(Linear(15, 1));model.add_next(AddBias(1));model.add_next(Sigmoid())
    sgd(dataset[:, :2], dataset[:, 2:],
    #sgd(np.array([[0.05, 0.10]]),
    #    np.array([[0.01, 0.99]]),
        model,
        loss,
        batch_size=32,
    #    batch_size=1,
        epochs=250,
        shuffle=True,
        lr=LEARNING_RATE)
    # TESTING
    #"""
    x = np.arange(0, 1, 1/100)
    y = np.arange(0, 1, 1/100)
    a = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    y_test = a.copy().T
    layer = model.next
    while layer is not None:
       y_test = layer.forward(y_test)
       layer = layer.next
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
    #"""
