import numpy as np
import math
from sklearn.model_selection import train_test_split
# 数学期望
def mean(x):
    return sum(x) / float(len(x))

# 标准差（方差）
def stdev( X):
    avg = mean(X)
    return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

# 概率密度函数
def gaussian_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2)/(2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# 处理X_train
def summarize(train_data):
    summaries = [(mean(i), stdev(i)) for i in zip(train_data)]
    return summaries

# 分类别求出数学期望和标准差
def fit(X, y):
    labels = list(set(y))
    data = {label: [] for label in labels}
    for f, label in zip(X, y):
        data[label].append(f)
    model = {label: summarize(value)
        for label, value in data.items()
    }
    return model

# 计算概率
def calculate_probabilities(model,input_data):
    # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
    # input_data:[1.1, 2.2]
    probabilities = {}
    for label, value in model.items():
        probabilities[label] = 1
        for i in range(len(value)):
            mean, stdev = value[i]
            probabilities[label] *= gaussian_probability(
                input_data[i], mean, stdev)
    return probabilities

# 类别
def predict(X_test,input_data):
    # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
    label = sorted(calculate_probabilities(X_test,input_data).items(),
        key=lambda x: x[-1])[-1][0]
    return label

def score(X_test, y_test):
    right = 0
    for X, y in zip(X_test, y_test):
        label = predict(X)
        if label == y:
            right += 1
    return right / float(len(X_test))

if __name__ == '__main__':
    data = np.loadtxt(open("path/test(100).csv", "rb"), delimiter=",", skiprows=0)
    x, y = np.split(data, (100,), axis=1)
    y = np.array([1 if i == 1 else -1 for i in y])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = fit(X_train,y_train)
    pre = predict(X_test,model)
    print('this is prediction')
    print(pre)