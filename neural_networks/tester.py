import numpy as np
import pandas as pd
from file_path import NN_FILE_PATH

x = np.array([[0.5, 0.5],[0,2],[-3,0.5]]).T
w = np.array([[1,0,-1,0],[0,1,0,-1]])
w_0 = np.array([[-1],[-1],[-1],[-1]])
w2 = np.array([[1,-1],[1,-1],[1,-1],[1,-1]])
w2_0 = np.array([[0],[2]])
y = np.array([[0, 1, 0]]).T

def z(w, w_0, x):
    return w.T@x + w_0

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    return np.exp(z)/sum(np.exp(z))

def nll(a,x,y):
    # print(x.shape)
    # print(a.T.shape)
    return x@(a - y).T

def backprop(w, dw, step_size):
    return w - step_size * dw


z_1 = z(w,w_0, x)
print(z_1)
a_1 = relu(z_1)
print(a_1)
# z_2 = z(w2,w2_0,a_1)
# print(z_2)
# a_2 = softmax(z_2)
# print(a_2)
# dw = nll(a,x,y)
# print(dw)
# new_w = backprop(w, dw, 0.5)
# print(new_w)