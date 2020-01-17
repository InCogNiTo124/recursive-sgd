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

    def forward(self, X):
        X = X.T
        layer = self.next
        while layer is not None:
            X = layer.forward(X)
            layer = layer.next
        return X

    def __call__(self, X):
        return self.forward(X)

