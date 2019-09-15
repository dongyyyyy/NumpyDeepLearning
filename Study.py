import numpy as np

a = np.array([[1,2,3,4],[5,6,7,8]])
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
b = np.array([1])
for i in range(len(a)):
    c = np.concatenate((b,a[i]),axis=0)
    print(c)
C_Y = np.concatenate((A, B), axis = 0)
print(C_Y)
C_X = np.concatenate((A, B), axis = 1)
print(C_X)