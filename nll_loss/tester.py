import numpy as np
import pandas as pd
from file_path import NLL_FILE_PATH
pengions = pd.read_csv(f"{NLL_FILE_PATH}/penguins.csv") # read in penguins
pengions = pengions.drop(["rowid","island","sex","year"], axis=1).dropna() #select data columns, and drop NAs
pengions = pengions.loc[pengions["species"].isin(["Adelie","Gentoo"])] # Want 2 pengion species to binary classify
pengions["bill_length_mm"] = (pengions["bill_length_mm"] - pengions["bill_length_mm"].mean() ) / pengions["bill_length_mm"].std() # standardize feature rows with mean 0 std 1 
pengions["bill_depth_mm"] = (pengions["bill_depth_mm"] - pengions["bill_depth_mm"].mean() ) / pengions["bill_depth_mm"].std()
pengions["flipper_length_mm"] = (pengions["flipper_length_mm"] - pengions["flipper_length_mm"].mean() ) / pengions["flipper_length_mm"].std()
pengions["body_mass_g"] = (pengions["body_mass_g"] - pengions["body_mass_g"].mean() ) / pengions["body_mass_g"].std()
pengions = pengions.sample(frac=1) # Mix up the datapoints so not all Adelie, then all Gentoo
labels = pengions["species"].to_numpy().reshape((1,len(pengions))) #y
labels = np.where(labels == "Adelie", 0, 1) # turn y values into 0, 1
data = pengions.drop(["species"], axis=1).to_numpy().T #x
# print(pengions.head())
# print(labels.head)
# print(data.head())
# print(pengions["species"].values)
# print(labels[:, 0])
# print(data[:, 0])

# X = d x n
# Y = 1 x n
def gd(X, Y):
    d = X.shape[0]
    th = np.zeros((d,1))
    th0 = np.zeros((1,1))
    lam = 0.0001
    # max_iters = 4
    def step_size_fn(i):
       return 2/(i+1)**0.1
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
        th = old_th - step_size_fn(t) * d_lr_obj_th(X, Y, old_th, old_th0, lam)
        th0 = old_th0 - step_size_fn(t) * d_lr_obj_th0(X, Y, old_th, old_th0)
        # print(th)
        # print(th0)

        if abs(lr_obj(X, Y, th, th0, lam) - lr_obj(X, Y, old_th, old_th0, lam)) < 0.00001:
        # if abs(lr_obj(X, Y, th, th0, lam) - lr_obj(X, Y, old_th, old_th0, lam)) < 1:
            # Print predictions
            # print(sigmoid(np.dot(th.T,X) + th0))
            break
        # t = t + 1
    return th, th0

# z = scalar or np array
def sigmoid(z):
    return 1/(1+np.e**(-z))

# actual_y = scalar / array (1 x n)
# pred_y = scalar / array (1 x n)
# output = scalar / 1 x n
def nll_loss(pred_y, actual_y):
    # print(actual_y)
    # print(pred_y)
    # print(-1*(actual_y * np.log(pred_y)))
    # print((1-actual_y) * np.log(1 - pred_y))
    # print(-1*(actual_y * np.log(pred_y) + (1-actual_y) * np.log(1 - pred_y)))
    # print(1/0)
    return -1*(actual_y * np.log(pred_y) + (1-actual_y) * np.log(1 - pred_y))

# X = d x n
# Y = 1 x n
# th = d x 1
# th0 = scalar
# lam = scalar
# output = 1 x n derivative showing the average rate of increase of the objective function as a function of
# th at th,th0 across all the data points
def lr_obj(X, Y, th, th0, lam):
    n = X.shape[1]
    mag_sqr = np.linalg.norm(th)**2
    # sigmoid --> 1 x n
    # nll_loss(1 x n, 1 x n) --> 1 x n
    # sum --> scalar
    # scalar + scalar = scalar
    return np.sum(nll_loss(sigmoid(np.dot(th.T,X) + th0),Y))/n + (lam/2)*mag_sqr

# Returns the gradient of logistic regression(x, y, th, th0) with respect to th
# X = d x n
# Y = 1 x n
# th = d x 1
# th0 = scalar
# lam = scalar
# output = 1 x n derivative showing the average rate of increase of the objective function as a function of
# th at th,th0 across all the data points
def d_lr_obj_th(X, Y, th, th0, lam):
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
    return np.sum((sigmoid(np.dot(th.T,X) + th0) - Y)*X, axis=1, keepdims=True)/n + lam*th

# print(np.array([1,2]) * np.array([[1,2],[1,2]]) - np.array([[1],[1]]))

# Returns the gradient of logistic regression(x, y, th, th0) with respect to th0
# X = d x n
# Y = 1 x n
# th = d x 1
# th0 = scalar
# output = 1 x 1 derivative showing the average rate of increase of the objective function at th,th0
# as a function of th0 across all the data points
def d_lr_obj_th0(X, Y, th, th0):
    n = X.shape[1]
    return np.sum(sigmoid(np.dot(th.T,X) + th0) - Y)/n

# x = d x n
# th = d x 1
# th0 = scalar
def hyperplane_side(x, th, th0):
    return np.sign(th.T@x + th0)

# data = d x n
# labels = 1 x n
# th = d x 1
# th0 = scalar
def score(data, labels, th, th0):
    ys = hyperplane_side(data,th,th0) # create predictions
    ys = np.where(ys == -1, 0, 1) # Change -1s to 0s as in NLL loss y-values in {0,1}
    bools = np.where(ys == labels, True, False) # see which are True/correct
    return np.sum(bools, axis=1)[0]/data.shape[1] # create percentage correct

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
        print(f"Score on iteration {i}: {score}")
        print("###################")
        mean_score = mean_score + score
    return mean_score / k

print(xval_learning_alg(gd, data, labels, 8))