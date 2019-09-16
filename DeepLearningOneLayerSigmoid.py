import numpy as np
from math import exp
from DeepLearningFunction import *

learning_rate = 0.0001


def V_Sigmoid():
    sigmoid = np.vectorize(Sigmoid)
    return sigmoid


def V_ReLU():
    relu = np.vectorize(ReLU)
    return relu


def Sigmoid(x):
    try:
        return 1 / (1 + exp(-x))
    except OverflowError:
        return 1.


def ReLU(x):
    return max(0, x)


if __name__ == "__main__":

    batch = 100
    epoch = 10
    startNumber = 0

    data = np.loadtxt('TrainDataset.csv', delimiter=',', dtype=np.float32)
    train_x_data = data[:, 0:-1]
    bias = [[1.] * 1 for i in range(len(train_x_data))]
    train_x_data_bias = np.concatenate((bias, train_x_data), axis=1)
    train_y_data = data[:, [-1]]
    train_y_data_onehot = []
    print("데이터 총 개수 : ", len(train_y_data))
    for i in range(len(train_y_data)):
        if train_y_data[i] == 1:
            train_y_data_onehot.append([1, 0, 0, 0, 0, 0, 0, 0])
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
    W1 = np.random.rand(4, 24)
    W2 = np.random.rand(24, 8)
    v_sigmoid = V_Sigmoid()
    v_relu = V_ReLU()


    # h1 = np.array(np.zeros(24))

    def forward(x, y):
        z1 = np.dot(x, W1)
        h1 = MakeFristOne(z1)

        z2 = np.dot(h1, W2)
        o = ODivideFunction(np.exp(z2), np.sum(np.exp(z2), axis=1))
        e = np.mean(-np.sum(y * np.log(o), axis=1))

        return e, o, h1


    def backward(x, y, W1, W2, input_data, h1):  # x = O(n) , y = label , W = Weight
        local_param2 = (x - y)
        result = HandFunction(h1, local_param2)
        delta_o = (-learning_rate) * result
        NW2 = W2 + delta_o

        '''
        sig2 = x*(1-x) # sigmoid' = O(n)[1-O(n)]
        local_param2 = (y-x)*sig2 # 출력노드 국부적기울기
        result = HandFunction(h1, local_param2)
        delta_o = (-learning_rate)*result
        NW2 = W2 + delta_o
        '''
        local_param1 = np.matmul(local_param2,np.transpose(W2)) # (24,8) (100 8) -> (100 24)
        result = HandFunction(input_data, local_param1) # (100,4), (100,24)
        delta_h1 = (-learning_rate) * result # (4,24)
        NW1 = W1 + delta_h1
        return NW1, NW2


    def Accuracy(x, y, batch):
        z1 = np.dot(x, W1)
        h1 = MakeFristOne(z1)

        z2 = np.dot(h1, W2)
        o = ODivideFunction(np.exp(z2), np.sum(np.exp(z2), axis=1))
        accuracy = 0.
        for i in range(batch):
            if (np.argmax(o[i]) == np.argmax(y[i])):
                accuracy = accuracy + 1.
        return accuracy / batch * 100, o


    maxBatch = int(len(train_x_data_bias) / batch)
    print("batch size   = ", batch)
    print("batch Number = ", maxBatch)
    count = 0
    for i in range(epoch):  # 10 번 반복
        Eavg = 0.
        startNumber = 0
        for j in range(maxBatch):  # 200번 반복
            x_batch = train_x_data_bias[startNumber:startNumber + 100]
            y_batch = train_y_data_onehot[startNumber:startNumber + 100]
            if (len(x_batch) != 0):
                Eav, error, h1 = forward(x_batch, y_batch)
                W1, W2 = backward(error, y_batch, W1, W2, x_batch, h1)
                # print(W1)
                Eavg = Eavg + Eav
                startNumber = startNumber + 100
        print("Epoch ", i + 1, "Eavg : ", Eavg / maxBatch)

    test = np.loadtxt('TestDataset.csv', delimiter=',', dtype=np.float32)
    test_x_data = data[:, 0:-1]
    test_bias = [[1.] * 1 for i in range(len(test_x_data))]
    test_x_data_bias = np.concatenate((test_bias, train_x_data), axis=1)
    test_y_data = data[:, [-1]]
    test_y_data_onehot = []
    for i in range(len(test_y_data)):
        if test_y_data[i] == 1:
            test_y_data_onehot.append([1, 0, 0, 0, 0, 0, 0, 0])
        elif test_y_data[i] == 2:
            test_y_data_onehot.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif test_y_data[i] == 3:
            test_y_data_onehot.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif test_y_data[i] == 4:
            test_y_data_onehot.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif test_y_data[i] == 5:
            test_y_data_onehot.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif test_y_data[i] == 6:
            test_y_data_onehot.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif test_y_data[i] == 7:
            test_y_data_onehot.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif test_y_data[i] == 8:
            test_y_data_onehot.append([0, 0, 0, 0, 0, 0, 0, 1])

    test = np.loadtxt('TestDataset.csv', delimiter=',', dtype=np.float32)
    test_x_data = data[:, 0:-1]
    test_bias = [[1.] * 1 for i in range(len(test_x_data))]
    test_x_data_bias = np.concatenate((test_bias, train_x_data), axis=1)
    test_y_data = data[:, [-1]]
    test_y_data_onehot = []
    for i in range(len(test_y_data)):
        if test_y_data[i] == 1:
            test_y_data_onehot.append([1, 0, 0, 0, 0, 0, 0, 0])
        elif test_y_data[i] == 2:
            test_y_data_onehot.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif test_y_data[i] == 3:
            test_y_data_onehot.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif test_y_data[i] == 4:
            test_y_data_onehot.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif test_y_data[i] == 5:
            test_y_data_onehot.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif test_y_data[i] == 6:
            test_y_data_onehot.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif test_y_data[i] == 7:
            test_y_data_onehot.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif test_y_data[i] == 8:
            test_y_data_onehot.append([0, 0, 0, 0, 0, 0, 0, 1])

    Eavg = 0.
    startNumber = 0
    Aavg = 0.
    for i in range(len(test_x_data_bias)):  # 200번 반복
        x_batch = test_x_data_bias[startNumber:startNumber + batch]
        y_batch = test_y_data_onehot[startNumber:startNumber + batch]
        if (len(x_batch) != 0):
            accuracy, pred = Accuracy(x_batch, y_batch, batch)
            print("Label : ", np.argmax(y_batch, axis=1))
            print("pred : ", np.argmax(pred, axis=1))
            Aavg = Aavg + accuracy
            startNumber = startNumber + 100
    print("Aavg : {}%".format((Aavg / len(test_x_data_bias))*100))
