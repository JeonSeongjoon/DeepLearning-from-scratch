import pdb
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(x):
    """
    Softmax function
    배치 처리를 올바르게 수행
    """
    
    C = np.max(x, axis=-1, keepdims=True)   # to prevent the overflow
    exp_x = np.exp(x - C)
    exp_sum = np.sum(exp_x, axis=-1, keepdims=True)
    sftmx_x = exp_x / exp_sum
    
    return sftmx_x

def crossEntropyLoss(label, pred, delta=1e-7):
    label = np.argmax(label, axis=1)
    batch_size = pred.shape[0]
   
    return (-1/batch_size) * np.sum(np.log(pred[np.arange(batch_size), label] + delta))


