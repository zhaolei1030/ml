from math import exp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + exp(-x))

def data_matrix(X):
    data_mat = []
    for d in X:
        data_mat.append([1.0, *d])
    return data_mat

def fit(x, y,max_iter,learning_rate):
    # label = np.mat(y)
    data_mat = data_matrix(X)  # m*n
    weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

    for iter_ in range(max_iter):
        for i in range(len(X)):
            result = sigmoid(np.dot(data_mat[i], weights))
            error = y[i] - result
            weights += learning_rate * error * np.transpose(
                [data_mat[i]])
    print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
        learning_rate, max_iter))
    return weights

def score(X_test, y_test,weights):
    right = 0
    X_test = data_matrix(X_test)
    for x, y in zip(X_test, y_test):
        result = np.dot(x, weights)
        if (result > 0 and y == 1) or (result < 0 and y == 0):
            right += 1
    return right / len(X_test)

if __name__ == "__main__":
    data = np.loadtxt(open("path/test(100).csv", "rb"), delimiter=",", skiprows=0)
    x, y = np.split(data, (100,), axis=1)
    y = np.array([1 if i == 1 else -1 for i in y])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    weights = fit(X_train, y_train, max_iter = 50, learning_rate = 0.6)
    result = score(X_test,y_test,weights)
