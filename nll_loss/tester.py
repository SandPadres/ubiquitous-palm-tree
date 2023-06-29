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
    max_iter = 50
    def step_size_fn(i):
       return 2/(i+1)**0.1

    for t in range(max_iter):
        old_th = th
        old_th0 = th0
        th = old_th - step_size_fn(t) * d_lr_obj_th(X, Y, old_th, old_th0, lam)
        th0 = old_th0 - step_size_fn(t) * d_lr_obj_th0(X, Y, old_th, old_th0)
    return th, th0

# z = scalar or np array
def sigmoid(z):
    return 1/(1+np.e**(-z))

# Returns the gradient of logistic regression(x, y, th, th0) with respect to th
# X = d x n
# Y = 1 x n
# th = d x 1
# th0 = scalar
# lam = scalar
# output = d x n gradient computed at th for all n data points
def d_lr_obj_th(X, Y, th, th0, lam):
    n = X.shape[1]

    # 1 x n * d x n = d x n
    # d x n + d x 1 = d x n
    return np.sum((sigmoid(np.dot(th.T,X) + th0) - Y)*X + lam*th)/n

# print(np.array([1,2]) * np.array([[1,2],[1,2]]) - np.array([[1],[1]]))

# Returns the gradient of logistic regression(x, y, th, th0) with respect to th0
# X = d x n
# Y = 1 x n
# th = d x 1
# th0 = scalar
# output = 1 x n gradient computed at th0 for all n data points
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