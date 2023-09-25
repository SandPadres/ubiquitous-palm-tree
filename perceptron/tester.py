import numpy as np
import pandas as pd
from file_path import PERCEPTRON_FILE_PATH
pengions = pd.read_csv(f"{PERCEPTRON_FILE_PATH}/penguins.csv") # read in penguins
pengions = pengions.drop(["rowid","island","sex","year"], axis=1).dropna() #select data columns, and drop NAs
pengions = pengions.loc[pengions["species"].isin(["Adelie","Gentoo"])] # Want 2 pengion species to binary classify
pengions["bill_length_mm"] = (pengions["bill_length_mm"] - pengions["bill_length_mm"].mean() ) / pengions["bill_length_mm"].std() # standardize feature rows with mean 0 std 1 
pengions["bill_depth_mm"] = (pengions["bill_depth_mm"] - pengions["bill_depth_mm"].mean() ) / pengions["bill_depth_mm"].std()
pengions["flipper_length_mm"] = (pengions["flipper_length_mm"] - pengions["flipper_length_mm"].mean() ) / pengions["flipper_length_mm"].std()
pengions["body_mass_g"] = (pengions["body_mass_g"] - pengions["body_mass_g"].mean() ) / pengions["body_mass_g"].std()
pengions = pengions.sample(frac=1) # Mix up the datapoints so not all Adelie, then all Gentoo
labels = pengions["species"].to_numpy().reshape((1,len(pengions))) #y
labels = np.where(labels == "Adelie", -1, 1) # turn y values into 1, -1
data = pengions.drop(["species"], axis=1).to_numpy().T #x
# print(pengions.head())
# print(labels.head)
# print(data.head())
# print(pengions["species"].values)
# print(labels[:, 0])
# print(data[:, 0])

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
    bools = np.where(ys == labels, True, False) # see which are True/correct
    return np.sum(bools, axis=1)[0]/data.shape[1] # create percentage correct

# learner = our ML algorithm function
# data_train = d x n 
# labels_train = 1 x n
# data_test = d' x n
# labels_test = d' x n
def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    # print(th)
    # print(th0)
    np.save('penguin_th.npy', th)
    np.save('penguin_th0.npy', th0)
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

print(xval_learning_alg(perceptron, data, labels, 7))
penguin_th = np.load("penguin_th.npy")
penguin_th0 = np.load("penguin_th0.npy")