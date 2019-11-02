import numpy as np
import math


def __init__(self, prob):
    self.pro_A, self.pro_B, self.pro_C = prob

# E step
def pmf(pro_A, pro_B,pro_C,i):
    pro_1 = pro_A * math.pow(pro_B, data[i]) * math.pow(
        (1 - pro_B), 1 - data[i])
    pro_2 = (1 - pro_A) * math.pow(pro_C, data[i]) * math.pow(
        (1 - pro_C), 1 - data[i])
    return pro_1 / (pro_1 + pro_2)

# M step
def fit(pro_A, pro_B,pro_C,data):
    count = len(data)
    print('初始 prob:{}, {}, {}'.format(pro_A, pro_B,pro_C))
    for d in range(count):
        print(range(count))
        pmf_list = [pmf(pro_A, pro_B,pro_C,k) for k in range(count)]
        pro_A = 1 / count * sum(pmf_list)
        pro_B = sum([pmf_list[k] * data[k] for k in range(count)]) / sum([pmf_list[k] for k in range(count)])
        pro_C = sum([(1 - pmf_list[k]) * data[k]for k in range(count)]) / sum([(1 - pmf_list[k]) for k in range(count)])
        print('{}/{}  pro_a:{}, pro_b:{}, pro_c:{}'.format(
            d + 1, count, pro_A, pro_B, pro_C))
    return pro_A, pro_B, pro_C


if __name__ == '__main__':
    data = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    pro_A, pro_B, pro_C = 0.6, 0.4, 0.5
    fit(pro_A, pro_B, pro_C,data)
