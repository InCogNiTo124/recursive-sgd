import numpy as np

layer_list = [2, 3, 1]
layers = [np.random.randn(b, a) for a, b in zip(layer_list, layer_list[1:])]
print(layers)

