import sys
import csv
import numpy as np
import recursive_sgd as sgd
from recursive_sgd.model import Model
from recursive_sgd.layers import AddBias, Linear, ReLU, Sigmoid, Tanh
from recursive_sgd.losses import MSE, CE

MODEL = Model()
LOSSES = {'MSE': MSE(), 'CE': CE()}

class SGDConfig:
    def __init__(self):
        return

    def __repr__(self):
        return repr(self.__dict__)

def parse_train(arguments):
    config = SGDConfig()
    config.to_shuffle = True
    i = 0
    last_layer_size = 0
    while i < len(arguments):
        arg = arguments[i]
        if arg in ['-e', '--epochs']:
            i += 1
            config.epochs = int(arguments[i])
        elif arg in ['--batch-size']:
            i += 1
            config.batch_size = int(arguments[i])
        elif arg in ['-d', '--dataset']:
            i += 1
            filename = arguments[i]
            config.dataset = filename
        elif arg in ['-i', '--input-size']:
            i += 1
            config.input_size = int(arguments[i])
            last_layer_size = config.input_size
        elif arg in ['-l']:
            i += 1
            layer_size = int(arguments[i])
            MODEL.add_next(Linear(last_layer_size, layer_size))
            last_layer_size = layer_size
        elif arg in ['-b']:
            MODEL.add_next(AddBias(last_layer_size))
        elif arg in ['-s']:
            MODEL.add_next(Sigmoid())
        elif arg in ['-t']:
            MODEL.add_next(Tanh())
        elif arg in ['-r']:
            MODEL.add_next(ReLU())
        elif arg  in ['--loss']:
            i += 1
            config.loss = LOSSES[arguments[i]]
        elif arg in ['--shuffle']:
            config.to_shuffle = True
        elif arg in ['--no-shuffle']:
            config.to_shuffle = False
        elif arg in ['--lr']:
            i += 1
            config.lr = float(arguments[i])
        else:
            print('UNRECOGNIZED OPTION:', arg)
        i += 1
    return config

METRICS = {'accuracy': lambda y_true, y_pred: np.mean(y_true == y_pred)}

def parse_test(arguments):
    config = SGDConfig()
    i = 0
    while i < len(arguments):
        arg = arguments[i]
        if arg in ['-m', '--model']:
            i += 1
            config.model = arguments[i]
        elif arg in ['-d', '--dataset']:
            i += 1
            config.dataset = arguments[i]
        elif arg in ['--metrics']:
            i += 1
            config.metrics = METRICS[arguments[i]]
        elif arg in ['-i', '--input-size']:
            i += 1
            config.input_size = int(arguments[i])
        i += 1
    return config

def main():
    if sys.argv[1] == 'train':
        config = parse_train(sys.argv[2:])
        with open(config.dataset, 'r') as file:
            dataset = np.array(list(csv.reader(file, delimiter=',')), dtype=np.float64)
            X_train = dataset[:, :config.input_size]
            y_train = dataset[:, config.input_size:]
        sgd.train(X_train, y_train, MODEL, config.loss, batch_size=config.batch_size,
                  epochs=config.epochs, shuffle=config.to_shuffle, lr=config.lr)
        sgd.save(MODEL, 'MODEL.sgd')
    if sys.argv[1] == 'test':
        config = parse_test(sys.argv[2:])
        model = sgd.load(config.model)
        with open(config.dataset, 'r') as file:
            dataset = np.array(list(csv.reader(file, delimiter=',')), dtype=np.float64)
            X_test = dataset[:, :config.input_size]
            y_test = dataset[:, config.input_size:]
        print(config.metrics(y_test.flatten(), (model.forward(X_test).flatten() > 0.5).astype(np.float64)))

    return

if __name__ == '__main__':
    main()
