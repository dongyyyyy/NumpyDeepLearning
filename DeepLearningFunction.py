import numpy as np


def HandFunction(h, f):
    result = np.array(np.zeros([len(h[0]), len(f[0])]))
    h = h.sum(axis=0)
    f = f.sum(axis=0)
    for i in range(len(h)):
        for j in range(len(f)):
            result[i][j] = h[i] * f[j]
    return result


def HandFunction1(h, f):
    result = np.array(np.zeros([len(f), len(f[0])]))
    for i in range(len(f)):  # 100
        for j in range(len(f[0])):  # 24
            result[i][j] = h[j] * f[i][j]  # 24 , (100,24) -> (100,24)
    return result


def HandFunction2(h, f):
    result = np.array(np.zeros([len(h), len(f[0])]))
    for i in range(len(f[0])):  # 24
        for j in range(len(h)):  # 100
            result[j][i] = h[j] * f[j][i]  # (100) , (100,24) -> (100,24)
    return result


def ODivideFunction(h,f):
    result = np.array(np.zeros([len(h), len(h[0])]))
    for i in range(len(h)): # 100
        for j in range(len(h[0])): # 8
            result[i][j] = h[i][j] / f[i]
    return result


def MakeFristOne(h):
    for i in range(len(h)):
        h[i][0] = 1.
    return h


def RetrunOneHot(input):
    if input == 0:
        return [1, 0, 0, 0, 0, 0, 0, 0]
    elif input == 1:
        return [0, 1, 0, 0, 0, 0, 0, 0]
    elif input == 2:
        return [0, 0, 1, 0, 0, 0, 0, 0]
    elif input == 3:
        return [0, 0, 0, 1, 0, 0, 0, 0]
    elif input == 4:
        return [0, 0, 0, 0, 1, 0, 0, 0]
    elif input == 5:
        return [0, 0, 0, 0, 0, 1, 0, 0]
    elif input == 6:
        return [0, 0, 0, 0, 0, 0, 1, 0]
    elif input == 7:
        return [0, 0, 0, 0, 0, 0, 0, 1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0]
