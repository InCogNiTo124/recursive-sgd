import numpy as np
LR = 0.5

def loss(y_pred, y_true):
    return np.mean((y_true - y_pred)**2)

def loss_grad(y_pred, y_true):
    return y_pred - y_true

def s(x):
    return 1 / (1 + np.exp(-x))

def s_grad(x):
    g = s(x)
    return g * (1 - g)

X = np.array([[0.05], [0.10]])
y = np.array([[0.01], [0.99]])

W1 = np.array([[0.15, 0.20], [0.25, 0.30]])
b1 = np.array([[0.35], [0.35]])
W2 = np.array([[0.40, 0.45], [0.50, 0.55]])
b2 = np.array([[0.60], [0.60]])

z1 = W1 @ X + b1
print("z1", z1)
a1 = s(z1)
print("a1", a1)

z2 = W2 @ a1 + b2
print("z2", z2)
a2 = s(z2)
print("a2", a2)

l = loss(a2, y)
print("loss", l)
grad = loss_grad(a2, y)
print("grad", grad)

