import math
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
def L(x, y, p=2):
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0

def predict(n,p,X_train,y_train,X):
    # 取出n个点
    knn_list = []
    for i in range(n):
        dist = np.linalg.norm(X - X_train[i], ord=p) #求范数
        knn_list.append((dist, y_train[i]))

    for i in range(n, len(X_train)):
        max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
        dist = np.linalg.norm(X - X_train[i], ord=p)
        if knn_list[max_index][0] > dist:
            knn_list[max_index] = (dist, y_train[i])

    # 统计
    knn = [k[-1] for k in knn_list]
    count_pairs = Counter(knn)
#         max_count = sorted(count_pairs, key=lambda x: x)[-1]
    max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
    return max_count

def score(number,p,X_train,y_train,X,X_test, y_test):
    right_count = 0
    n = 10
    for X, y in zip(X_test, y_test):
        label = predict(number,p,X_train,y_train,X)
        if label == y:
            right_count += 1
    return right_count / len(X_test)

if __name__ == "__main__":
    data = np.loadtxt(open("path/test(100).csv", "rb"), delimiter=",", skiprows=0)
    x, y = np.split(data, (100,), axis=1)
    y = np.array([1 if i == 1 else -1 for i in y])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    score(3,2,X_train,y_train,X_test, X_test, y_test)