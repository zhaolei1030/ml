import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def sign(x, w, b):
    y = np.dot(x, w) + b
    return y

def fit(X_train, y_train):
    """
    随机梯度下降
    :param X_train:
    :param y_train:
    :return:
    """
    statue = False
    while not statue:
        count = 0
        for d in range(len(X_train)):
            X = X_train[d]
            y = y_train[d]
            if y * sign(X, w, b) <= 0:
                w = w + l_rate * np.dot(y, X)
                b = b + l_rate * y
                count += 1
        if count == 0:
            statue = True
    return "finish"

if __name__ == "__main__":
    data = np.loadtxt(open("path/test(100).csv", "rb"), delimiter=",", skiprows=0)
    x, y = np.split(data, (100,), axis=1)
    y = np.array([1 if i == 1 else -1 for i in y])
    w = np.ones(len(data[0]) - 1, dtype=np.float32)   #初始化
    b = 0
    l_rate = 0.1
    fit(x, y)
    x_line = np.linspace(4, 7, 10)
    y_line = -(w[0]*x_line + b)/w[1]
    plt.plot(x_line, y_line)
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('length')
    plt.ylabel('width')
    plt.legend()
