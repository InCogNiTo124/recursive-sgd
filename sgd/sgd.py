import numpy as np
import logging
log = logging.getLogger(__name__)

def sgd(X, y_true, layer_list, loss=None, batch_size=4, epochs=1, shuffle=False, lr=1e-3):
    for epoch in range(1, epochs+1):
        #print("EPOCH: {}".format(epoch))
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
    layer = layer_list
    if layer is None:
        log.debug("y_pred {}".format(X))
        log.debug("y_true {}".format(y))
        log.info("loss {}".format(loss.forward(X, y)))
        delta = loss.backward(X, y)
        log.debug("delta {}".format(delta))
        return delta
    else:
        a = layer.forward(X)
        grad = sgd_step(a, y, layer.next, loss, lr)
        log.debug(layer.__class__.__name__)
        log.debug("input {}".format(X))
        log.debug("output {}".format(a))
        log.debug("grad {}".format(grad))
        back = layer.backward(X)
        log.debug("back {}".format(back))
        n_grad = np.dot(back, grad)
        log.debug("n_grad {}".format(n_grad))
        if layer.trainable:
            layer.update(X, grad, lr)
            log.info("NEW VALUES {}".format(layer.get_weights()))
        return n_grad
        #dw = n_grad.dot(X.T)
        #db = np.mean(n_grad, axis=1, keepdims=True)
        #assert dw.shape == layer.lin.W.shape
        #L_grad = layer.lin.W.T @ n_grad
        #layer.lin.W -= LEARNING_RATE*dw
        #layer.lin.b -= LEARNING_RATE*db
        #return l_grad
