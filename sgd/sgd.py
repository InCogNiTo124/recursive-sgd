import numpy as np
import logging
log = logging.getLogger(__name__)

def sgd(X, y_true, layer_list, loss=None, batch_size=4, epochs=1, shuffle=False, lr=1e-3):
    """Trains a neural net, calling the recursive function.
    A neural net is trained in-place (no new object is created)

    :param X: np.ndarray, the features of the dataset
    :param y_true: np.ndarray, the vector of labels
    ;param layer_list:  sgd.Model, the model that will be trained in-place
    :param loss: sgd.Loss, a loss function to measure the error a batch of inputs makes for the corresponding outputs (Default value = None)
    :param batch_size: int, the number of examples in one batch (Default value = 4)
    :param epochs: int, the number of times the function will loop over the enitre dataset (Default value = 1)
    :param shuffle: bool, whether or not will the dataset be shuffled before every epoch (Default value = False)
    :param lr: float, the learning rate (Default value = 1e-3)
    :param layer_list: 
    :returns: None, the model was trained in-place and is available in the caller's scope

    """
    for epoch in range(1, epochs+1):
        #log.info("EPOCH: {}".format(epoch))
        if shuffle:
            indices = np.random.permutation(X.shape[0])
        else:
            indices = np.arange(X.shape[0])
        for i in range(X.shape[0]//batch_size):
            start = i*batch_size
            end = start + batch_size
            X_batch = X[indices[start:end], :]
            y_batch = y_true[indices[start:end], :]
            sgd_step(X_batch.T, y_batch.T, layer_list.next, loss, lr)
    return

def sgd_step(X, y, layer_list, loss, lr):
    """
    Does one step of the gradient descent. Calls itself recursively.
    The idea is to pass the input through one layer and let the function train
    the layers behind. The function recieves the delta vector of gradients from
    the next layers and calculates the gradients of weights and biases. It then
    updates those parameters and, at last, returns the delta vector of the
    current layer to the previous.

    :param X:           np.ndarray, the input matrix 
    :param y:           np.ndarray, the label matrix
    :param layer_list:  sgd.Model, the neural network model
    :param loss:        sgd.Loss, the loss function
    :param lr:          float, the learning rate
    :returns:           np.ndarray, the delta vector over inputs
    """
    layer = layer_list
    if layer is None:
        log.debug("y_pred {}".format(X))
        log.debug("y_true {}".format(y))
        log.info("loss {}".format(loss.forward(X, y)))
        delta = loss.backward(X, y)
        log.debug("delta {}".format(delta))
        return delta
    else:
        log.debug(layer.__class__.__name__)
        log.debug("input {}".format(X))
        a = layer.forward(X)
        log.debug("output {}".format(a))
        delta = sgd_step(a, y, layer.next, loss, lr)
        log.debug("delta {}".format(delta))
        grad = layer.backward(X, delta)
        log.debug("grad \n{}".format(grad))
        if layer.trainable:
            layer.update(X, delta, lr)
            log.debug("NEW VALUES {}".format(layer.get_weights()))
        return grad

