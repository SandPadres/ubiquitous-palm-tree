import numpy as np
import pandas as pd
from file_path import REGRESSION_FILE_PATH
fish = pd.read_csv(f"{REGRESSION_FILE_PATH}/fish.csv") # read in fish
fish = fish.dropna() # drop NAs
fish["Length1"] = (fish["Length1"] - fish["Length1"].mean()) / fish["Length1"].std() # standardize feature rows with mean 0 std 1 
fish["Length2"] = (fish["Length2"] - fish["Length2"].mean()) / fish["Length2"].std()
fish["Length3"] = (fish["Length3"] - fish["Length3"].mean()) / fish["Length3"].std()
fish["Height"] = (fish["Height"] - fish["Height"].mean()) / fish["Height"].std()
fish["Width"] = (fish["Width"] - fish["Width"].mean()) / fish["Width"].std()
fish["Weight"] = (fish["Weight"] - fish["Weight"].mean()) / fish["Weight"].std() # standardize the y-values with mean 0 and variance 1
Y_STD = fish["Weight"].std()
species_list = list(fish['Species'].unique()) # one hot encode the Species column
# v = int
# entries = list of all the options
def one_hot_encode(x, entries):
    '''
    Outputs a one hot vector. Helper function to be used in auto_data_and_labels.
    v is the index of the "1" in the one-hot vector.
    entries is range(k) where k is the length of the desired onehot vector. 

    >>> one_hot(2, range(4))
    [0, 0, 1, 0]
    >>> one_hot(1, range(5))
    [0, 1, 0, 0, 0]
    '''
    vec = len(entries)*[0]
    vec[entries.index(x)] = 1
    return vec
fish['Species'] = fish['Species'].apply(lambda x: one_hot_encode(x,species_list))
# print(fish.head)
# df2 = pd.DataFrame(fish['Species'].str.split().values.tolist())
# print(df2.head)
# fish[['team1','team2']] = pd.DataFrame(df2.teams.tolist(), index= df2.index)
fish = fish.sample(frac=1) # Mix up the datapoints so not fish of the same type are next to each other
labels = fish["Weight"].to_numpy().reshape((1,len(fish))) #y
data_species = np.array(fish.Species.tolist()).T #x
data_rest = fish.drop(["Weight","Species"], axis=1).to_numpy().T
data = np.vstack([data_species, data_rest])
# print(species_list)
# print(len(fish))
# print(fish.head())
# print(labels[:, 0])
# print(data[:, 0])
# print(data.shape)
# print(labels.shape)

# X = d x n
# Y = 1 x n
def gd(X, Y):
    d = X.shape[0]
    th = np.zeros((d,1))
    th0 = np.zeros((1,1))
    lam = 0.000
    max_iters = 200
    def step_size_fn(i):
       return 2/(i+1)**4
    t = 0

    # for t in range(max_iters):
    while True:
        old_th = th
        old_th0 = th0
        # print(X.shape)
        # print(Y.shape)
        # print(old_th.shape)
        # print(old_th0.shape)
        # print(lam)
        th = old_th - step_size_fn(t) * d_ms_obj_th(X, Y, old_th, old_th0, lam)
        th0 = old_th0 - step_size_fn(t) * d_ms_obj_th0(X, Y, old_th, old_th0)
        # print(th)
        # print(th0)

        if abs(ms_obj(X, Y, th, th0, lam) - ms_obj(X, Y, old_th, old_th0, lam)) < 0.00001:
            break
        t = t + 1
    return th, th0

# X = d x n
# Y = 1 x n
# th = d x 1
# th0 = scalar
# lam = scalar
# scalar = Value of the optimization function using th and th0
def ms_obj(X, Y, th, th0, lam):
    n = X.shape[1]
    mag_sqr = np.linalg.norm(th)**2

    # 1 x n --> sum --> scalar 
    # scalar + scalar = scalar
    return np.sum((np.dot(th.T,X) + th0 - Y)**2)/n + (lam/2)*mag_sqr

# Returns the gradient of logistic regression(x, y, th, th0) with respect to th
# X = d x n
# Y = 1 x n
# th = d x 1
# th0 = scalar
# lam = scalar
# output = 1 x n derivative showing the average rate of increase of the objective function as a function of
# th at th,th0 across all the data points
def d_ms_obj_th(X, Y, th, th0, lam):
    n = X.shape[1]
    # print(X.shape[0])
    # print(X.shape[1])

    # 1 x n * d x n = d x n --> sum --> d x 1
    # d x 1 + d x 1 = d x 1
    # print(th.T.shape)
    # print(X.shape)
    # print((np.dot(th.T,X) + th0).shape)
    # print((sigmoid(np.dot(th.T,X) + th0) - Y).shape)
    # print((sigmoid(np.dot(th.T,X) + th0) - Y).shape)
    # print(np.sum((sigmoid(np.dot(th.T,X) + th0) - Y)*X, axis=1, keepdims=True).shape)
    # print((np.sum((sigmoid(np.dot(th.T,X) + th0) - Y)*X, axis=1, keepdims=True)/n + lam*th).shape)
    # print(1/0)
    # print((sigmoid(np.dot(th.T,X) + th0) - Y).shape)
    return 2 * np.sum((np.dot(th.T,X) + th0 - Y)*X, axis=1, keepdims=True)/n + lam*th

# print(np.array([1,2]) * np.array([[1,2],[1,2]]) - np.array([[1],[1]]))

# Returns the gradient of logistic regression(x, y, th, th0) with respect to th0
# X = d x n
# Y = 1 x n
# th = d x 1
# th0 = scalar
# output = 1 x 1 derivative showing the average rate of increase of the objective function at th,th0
# as a function of th0 across all the data points
def d_ms_obj_th0(X, Y, th, th0):
    n = X.shape[1]
    # sum --> 1x1
    return 2 * np.sum(np.dot(th.T,X) + th0 - Y)/n

# In all the following definitions:
# x is d by n : input data
# y is 1 by n : output regression values
# th is d by 1 : weights
# th0 is 1 by 1 or scalar
# output = 1 x n
def lin_reg(x, th, th0):
    """ Returns the predicted y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 0.]])
    >>> lin_reg(X, th, th0).tolist()
    [[1.05, 2.05, 3.05, 4.05]]
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> lin_reg(X, th, th0).tolist()
    [[3.05, 4.05, 5.05, 6.05]]
    """
    return np.dot(th.T, x) + th0

# x is d by n : input data
# y is 1 by n : output regression values
# th is d by 1 : weights
# th0 is 1 by 1 or scalar
# output = 1 x n
def square_loss(x, y, th, th0):
    """ Returns the squared loss between y_pred and y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> square_loss(X, Y, th, th0).tolist()
    [[4.2025, 3.4224999999999985, 5.0625, 3.8025000000000007]]
    """
    # print(lin_reg(x, th, th0))
    # print(y)
    # print(th)
    # print(lin_reg(x, th, th0)[:,:10])
    # print(y[:,:10])
    return (lin_reg(x, th, th0) - y)**2

# x is d by n : input data
# y is 1 by n : output regression values
# th is d by 1 : weights
# th0 is 1 by 1 or scalar
# output = 1 x 1, mean of all the square errors for all data points
def mean_square_loss(x, y, th, th0):
    """ Return the mean squared loss between y_pred and y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> mean_square_loss(X, Y, th, th0).tolist()
    [[4.1225]]
    """
    # the axis=1 and keepdims=True are important when x is a full matrix
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True)

# data = d x n
# labels = 1 x n
# th = d x 1
# th0 = scalar
# output = scalar, the RMSE (root mean squared error) on data (test), Y (test) * Y standard deviation
def score(data, labels, th, th0):
    return np.sqrt(mean_square_loss(data, labels, th, th0)).item() * Y_STD

# learner = our ML algorithm function
# data_train = d x n 
# labels_train = 1 x n
# data_test = d' x n
# labels_test = d' x n
def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    return score(data_test, labels_test, th, th0)

# learner = our ML algorithm
# data = d x n
# labels = 1 x n
# k = scalar
def xval_learning_alg(learner, data, labels, k):
    #cross validation of learning algorithm
    data_parts = np.array_split(data, k, axis = 1) # split data into k parts
    label_parts = np.array_split(labels, k, axis = 1) # split labels into k parts
    mean_score = 0
    for i in range(k):
        data_train = data_parts[:i] + data_parts[i+1:] # all but kth part
        data_train = np.concatenate((data_train),axis = 1)
        labels_train = label_parts[:i] + label_parts[i+1:] # all but kth part
        labels_train = np.concatenate((labels_train),axis = 1)
        data_test = data_parts[i] # kth part
        labels_test = label_parts[i] # kth part
        score = eval_classifier(learner, data_train, labels_train, data_test, labels_test)
        print(f"RMSE on iteration {i}: {score}")
        print("###################")
        mean_score = mean_score + score
    return mean_score / k

print(xval_learning_alg(gd, data, labels, 8))