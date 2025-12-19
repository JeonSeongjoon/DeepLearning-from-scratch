import numpy as np
from collections import OrderedDict

from computational_graph import Affine, ReLU, SoftmaxWithLoss


class TwoLayerNN:
    """
    I used this model for classification at first of this experiment.
    But, because of underfitting and several issues, I changed this 
    model to a single affine layer and a single softmax layer.
    """
    def __init__(self, input_size, output_size, hidden_size, weight_init_std=0.01):
        rn = np.random.randn
        zr = np.zeros

        W1 = weight_init_std * rn(input_size, hidden_size)
        W2 = weight_init_std * rn(hidden_size, output_size)
        b1 = zr((1, hidden_size))
        b2 = zr((1, output_size))
        self.params = [W1, b1, W2, b2]
        
        
        # I changed the codes to apply the optimizer more conveniently

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params[0], self.params[1])
        self.layers['ReLU'] = ReLU()
        self.layers['Affine2'] = Affine(self.params[2], self.params[3])
        self.last_layer = SoftmaxWithLoss() 

        self.Loss = None
        self.grads = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)]

    
    def forward(self, X, tgt):
        for layer in self.layers.values():
            X = layer.forward(X)
        loss = self.last_layer.forward(tgt, X)

        self.Loss = loss
        return loss


    def backward(self, dout=1):
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        self.grads[0][...] = self.layers['Affine1'].dW
        self.grads[1][...] = self.layers['Affine1'].db
        self.grads[2][...] = self.layers['Affine2'].dW
        self.grads[3][...] = self.layers['Affine2'].db

        return dout
    

class Single_Affine_Soft:
    """
    Changed the "TwoLayerNN" to this one
    """
    def __init__(self, input_size, output_size, weight_init_std=0.01):
        rn = np.random.randn
        zr = np.zeros

        W1 = rn(input_size, output_size) * np.sqrt(2.0 / input_size)
        b1 = zr(output_size) 
        self.params = [W1, b1]

        self.Affine= Affine(self.params[0], self.params[1])
        self.Soft = SoftmaxWithLoss() 

        self.Loss = None
        self.grads = [np.zeros_like(W1), np.zeros_like(b1)]

    
    def forward(self, X, tgt):
        X = self.Affine.forward(X)
        loss = self.Soft.forward(tgt, X)

        self.Loss = loss
        return loss


    def backward(self, dout=1):
        dout = self.Soft.backward(dout)
        dout = self.Affine.backward(dout)

        self.grads[0][...] = self.Affine.dW
        self.grads[1][...] = self.Affine.db

        return dout
    
   

class RNN_parts:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params

        t = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dtanh = dh_next * (1 - (h_next ** 2))
        
        db = np.sum(dtanh, axis=0)
        dWh = np.matmul(h_prev.T, dtanh)
        dh_prev = np.matmul(dtanh, Wh.T)
        dWx = np.matmul(x.T, dtanh)
        dx = np.matmul(dtanh, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev



class TimeRNN:
    """
    It is updated by backpropagation through time
    """
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        
        self.h = None                 # It contains the hidden state vector of the last RNN_parts
        self.dh = None                # It contains the gradient from the previous TimeRNN block
        self.stateful = stateful      # It represents the existence of the hidden state.

        self.vshape = None
        self.clip_thresh = 5.0
        self.truncate_length = 28        # For MNIST_ds, we don't need to truncate it 

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.vshape = (N, T, D)

        self.h_all = np.zeros((N, T + 1, H), dtype='f')
        self.layers = []

        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H), dtype = 'f')

        self.h_all[:, 0, :] = self.h

        for t in range(T):
            layer = RNN_parts(*self.params)
            self.layers.append(layer)

            self.h = layer.forward(xs[:, t, :], self.h)
            self.h_all[:, t + 1, :] = self.h

        logit = self.h_all[:, -1, :]
        return logit
        
    
    def backward(self, dh):                  
        N, T, D = self.vshape            
        self.dh = dh
        dxs = np.empty((N, T, D), dtype='f')

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dh)             # By the one to one structure, I just need the gradient from a next block.
            dxs[:, t, :] = dx                       # save the values of dx

            for i, grad in enumerate(layer.grads):  # grad => [dWx, dWh, db]
                grads[i] += grad

            #if (T - t) % self.truncate_length == 0: # truncated BPTT : reinitialize gradient as zero
            #    dh = 0                              
        

        for i in range(len(grads)):                 # Applying the gradient clipping
            np.clip(grads[i], -self.clip_thresh, self.clip_thresh, out=grads[i])
            

        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
        
        return dxs



class VanillaRNN:
    """
    Docstring for VanillaRNN
    """
    def __init__(self, input_size, hidden_size, output_size):
        D, H = input_size, hidden_size
        rn = np.random.randn
        zr = np.zeros
        
        # weight initialization 
        #>> I used the Xavier initialization
        rnn_Wx = rn(D, H) * np.sqrt(2.0 / D)
        rnn_Wh = rn(H, H) * np.sqrt(2.0 / H)
        rnn_b = zr(H)


        self.rnn_layer = TimeRNN(rnn_Wx, rnn_Wh, rnn_b)
        self.NN = Single_Affine_Soft(hidden_size, output_size)
        self.layers = [
            self.rnn_layer,
            self.NN
        ]         

        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, tgts):
        h_last = self.rnn_layer.forward(xs)         # just using the last one for many-to-one structure.
        loss = self.NN.forward(h_last, tgts)
        return loss
    
    def backward(self, dout=1):
        dout = self.NN.backward(dout)
        dxs = self.rnn_layer.backward(dout)
        return dxs
    
    def reset_state(self):
        self.rnn_layer.reset_state()


    