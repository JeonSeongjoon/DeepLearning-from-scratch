import numpy as np
import matplotlib.pyplot as plt 


def step_function(x):
    y = x > 0
    return y.astype(int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def I(x):
    return x

def Softmax(x):
    C = np.max(x)
    exp_x = np.exp(x-C)
    exp_sum = np.sum(exp_x)
    sftmx_x = exp_x / exp_sum
    return sftmx_x

def CrossEntropyLoss(label, pred, delta=1e-7):
    if pred.ndim == 1:
        label.reshape(1, label.size)
        pred.reshape(1, pred.size)

    batch_size = pred.size
    return (-1/batch_size) * np.sum(np.log(pred[np.arange(batch_size), label] + delta))




if __name__=="__main__":

    A = np.linspace(-10,10,100)
    res_A = ReLU(A)

    print(f"original value: {A[:5]}")
    print(f"after sigmoid: {res_A[:5]}")

    # plot the graph
    X = A
    Y = res_A

    plt.scatter(X, Y)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.show()