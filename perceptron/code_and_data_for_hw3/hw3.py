import numpy as np

# data = d x n
# labels = 1 x n
def perceptron_through_origin(data, labels):
    d = data.shape[0]
    n = data.shape[1]

    th = np.zeros((d,1))
    mistakes = 0
    correct = 0
    while correct != n:
        correct = 0
        for i in range(n):
            x_i = data[:,i:i+1] # Get data point
            y_i = labels[0,i] # Label for that data point
            if y_i * (np.dot(th.T,x_i)) <= 0:
                mistakes = mistakes + 1
                th = th + y_i * x_i
            else:
                correct = correct + 1
    return (th, mistakes)

# data = d x n
# labels = 1 x n
def perceptron(data, labels, params={}):
    # if T not in params, default to 100
    T = params.get('T', 100)
    d = data.shape[0]
    n = data.shape[1]

    th = np.zeros((d,1))
    th0 = np.zeros((1,1))
    for t in range(T):
        for i in range(n):
            x_i = data[:,i:i+1] # Get data point
            y_i = labels[0,i] # Label for that data point
            if y_i * (np.dot(th.T,x_i) + th0) <= 0:
                th = th + y_i * x_i
                th0 = th0 + y_i
    return (th, th0)

# x = scalar
# k = scalar
def one_hot(x, k):
    a = np.zeros((k,1))
    a[x-1,0] = 1
    # print(a)
    return a

# data = np.array([[0.2, 0.8, 0.2, 0.8],[0.2,  0.2,  0.8,  0.8],[1, 1, 1, 1]])
# ds = [2, 3,  4,  5]
ds = [1, 2, 3, 4, 5, 6]
data = np.empty((6, 0))
for d in ds:
    data = np.append(data, one_hot(d,6), axis = 1)
# print(data)
# labels = np.array([[-1, -1, 1, 1]])
# labels = np.array([[1, 1, -1, -1]])
labels = np.array([[1, 1, -1, -1, 1, 1]])
print(perceptron(data,labels))