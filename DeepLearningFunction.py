import numpy as np

def HandFunction(h,f,batch):
    result = np.array(np.zeros([len(h[0]), len(f[0])]))
    h = h.sum(axis=0)
    f = f.sum(axis=0)
    for i in range(len(h)):
        for j in range(len(f)):
            result[i][j] = h[i] * f[j]
    return result

def HandFunction1(h,f,batch):
    result = np.array(np.zeros([len(f),len(f[0])]))
    for i in range(len(f)):
        for j in range(len(f[0])):
            result[i][j] = h[j] * f[i][j] # 100, ( 100,96 ) 192 , (96,192)
    return result

def HandFunction2(h,f,batch):
    result = np.array(np.zeros([len(h),len(f[0])]))
    for i in range(len(f[0])):
        for j in range(len(h)):
            result[j][i] = h[j] * f[j][i] # 100, ( 100,96 ) 192 , (96,192)
    return result


def MakeFristOne(h):
    for i in range(len(h)):
        h[i][0] = 1.
    return h