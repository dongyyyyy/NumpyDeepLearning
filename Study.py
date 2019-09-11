import numpy as np

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b = np.array([[1],[2],[3]])

print(a*b) # 1을 1행에 모두 곱 2를 2행에 모두 곱

c = np.array([1,2,3,4,5,6,7,8,9,10]) # 인풋 데이터
d = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]) # 국부적 기울기

result = np.array(np.zeros([len(c),len(d)]))

print(result.shape)
for i in range(len(c)):
    for j in range(len(d)):
        result[i][j] = c[i]*d[j]

np.

print(result)
