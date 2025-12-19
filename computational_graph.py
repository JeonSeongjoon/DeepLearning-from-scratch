import numpy as np

from cost_fn import crossEntropyLoss, softmax, sigmoid

class MulLayer:
    def __init__(self):
        """
        save some values for the backward propagation process
        """
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x          
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = self.out * (1 - self.out) * dout
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.tgt = None

    def forward(self, x):
        self.x = x
        out = np.matmul(x, self.W) + self.b
        return out

    def backward(self, dout=1):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.tgt = None

    def forward(self, tgt, out):
        self.tgt = tgt
        self.y = softmax(out)
        self.loss = crossEntropyLoss(self.tgt, self.y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.tgt.shape[0]
        dx = (self.y - self.tgt) / batch_size
        return dx
