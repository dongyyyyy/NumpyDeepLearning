import numpy as np
from math import exp

learning_rate = 0.001

def V_Sigmoid():
    sigmoid = np.vectorize(Sigmoid)
    return sigmoid

def Sigmoid(x):
    return 1/(1 + exp(-x))
def ReLU(x):
    return max(0,x)

if __name__ == "__main__":

    data = np.loadtxt('TrainDataset.csv',delimiter=',',dtype=np.float32)
    train_x_data = data[:,0:-1]
    train_y_data = data[:,[-1]]
    train_y_data_onehot = []
    print("데이터 총 개수 : ", len(train_y_data))
    for i in range(len(train_y_data)):
        if train_y_data[i] == 1:
            train_y_data_onehot.append([1,0,0,0,0,0,0,0])
        elif train_y_data[i] == 2:
            train_y_data_onehot.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif train_y_data[i] == 3:
            train_y_data_onehot.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif train_y_data[i] == 4:
            train_y_data_onehot.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif train_y_data[i] == 5:
            train_y_data_onehot.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif train_y_data[i] == 6:
            train_y_data_onehot.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif train_y_data[i] == 7:
            train_y_data_onehot.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif train_y_data[i] == 8:
            train_y_data_onehot.append([0, 0, 0, 0, 0, 0, 0, 1])


    W1 = np.random.rand(3,10) # X = 3 / L1의 노드의 수 = 10
    W2 = np.random.rand(10, 10)  # X = 3 / L1의 노드의 수 = 10
    W3 = np.random.rand(10, 10)  # X = 3 / L1의 노드의 수 = 10
    W4 = np.random.rand(10, 10)  # X = 3 / L1의 노드의 수 = 10
    W5 = np.random.rand(10, 8)  # X = 3 / L1의 노드의 수 = 10
    #print(W1)
    print(train_x_data[0].shape,W1.shape)
    z1 = np.dot(train_x_data[0],W1) # matmul과 dot의 차이점 공부할 것
    print(z1)
    print(z1.shape)
    v_sigmoid = V_Sigmoid()
    h1 = np.array(np.zeros(10))
    h2 = np.array(np.zeros(10))
    h3 = np.array(np.zeros(10))
    h4 = np.array(np.zeros(10))
    def forward(x,y):
        z1 = np.dot(x,W1)
        h1 = v_sigmoid(z1)
        z2 = np.dot(h1,W2)
        h2 = v_sigmoid(z2)
        z3 = np.dot(h2,W3)
        h3 = v_sigmoid(z3)
        print("h3",h3.shape)
        z4 = np.dot(h3,W4)
        print("W4 : ", W4)
        print("z4 : ",z4)
        h4 = v_sigmoid(z4)
        z5 = np.dot(h4,W5)
        print("z5 : ",z5)
        print("h4 : ",h4)
        print("W5 : ",W5)
        o = v_sigmoid(z5)
        print(o.shape)
        print("="*20)
        print(y)
        print("=" * 20)
        print("o : ",o)
        print("=" * 20)
        print(y-o)
        e = np.sum(np.square(y - o))/2
        print("=" * 20)
        print("e: ",e)
        return e , o

    def backward(x,y):
        print(x)
        sig5 = x*(1-x) # f`(x)
        print(sig5)
        w5 = (y-x)*sig5 # 국부적 기울기
        print("w5:",w5)
        print("h4:",h4)
        W5 = (-learning_rate)*w5*h4

    Eav, error = forward(train_x_data[0], train_y_data_onehot[0])
    backward(error,train_y_data_onehot[0])


