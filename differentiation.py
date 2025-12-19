import copy
import numpy as np


def numerical_gradient(f, x):
    '''
    The function that gets the gradient value of vector-valued funciton f at x

    f : A vector-valued function f
    x : A point x 
    '''

    tmp_x = copy.deepcopy(x)
    grad = np.zeros_like(x)
    h = 1e-4

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        hej = np.zeros_like(x) 
        hej[idx] = h

        fx1 = f(tmp_x + hej) 
        fx2 = f(tmp_x - hej)

        grad[idx] = (fx1-fx2)/(2*h)
        it.iternext()

    return grad



# nditer
x = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 배열

it = np.nditer(x, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index
    print(f"인덱스: {idx}, 값: {x[idx]}")
    it.iternext()