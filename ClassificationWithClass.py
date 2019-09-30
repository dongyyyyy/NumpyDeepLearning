import numpy as np
from math import exp
from DeepLearningFunction import *

learning_rate = 0.0001


class Model:
    def __init__(self, batch):
        self.W1 = np.random.rand(4,24)
        self.W2 = np.random.rand(24,8)
        self.h1 = np.random.rand(batch, 24)

    def forward(self, input_data, input_truth):
        z1 = np.dot(input_data, self.W1)
        self.h1 = MakeFristOne(z1)

        z2 = np.dot(self.h1, self.W2)
        o = ODivideFunction(np.exp(z2), np.sum(np.exp(z2),axis=1))
        e = np.mean(-np.sum(input_truth*np.log(o),axis=1))

        return e, o

    def backward(self, pred, truth, input_data): # x = O(n) , y = label , W = Weight
        local_param2 = (pred - truth)
        result = np.dot(np.transpose(self.h1), local_param2)
        delta_o = (-learning_rate) * result
        nw2 = self.W2 + delta_o

        local_param1 = np.dot(local_param2, np.transpose(self.W2))
        result = HandFunction(input_data, local_param1)
        delta_h1 = (-learning_rate) * result
        nw1 = self.W1 + delta_h1

        self.W1 = nw1
        self.W2 = nw2

    def Accuracy(self, input_data, input_truth, batch):
        z1 = np.dot(input_data, self.W1)
        h1 = MakeFristOne(z1)

        z2 = np.dot(h1, self.W2)
        o = ODivideFunction(np.exp(z2), np.sum(np.exp(z2), axis=1))
        accuracy = 0.
        for i in range(batch):
            if np.argmax(o[i]) == np.argmax(input_truth[i]):
                accuracy = accuracy + 1.

        return accuracy/batch * 100, o

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

    data = np.loadtxt('TrainDataset_03.csv', delimiter=',', dtype=np.float32)
    train_x_data_03 = data[:, 0:-1]
    bias = [[1.] * 1 for i in range(len(train_x_data_03))]
    train_x_data_03_bias = np.concatenate((bias, train_x_data_03), axis=1)
    train_y_data_03 = data[:, [-1]]
    train_y_data_03_onehot = []


    print("데이터 총 개수 : ", len(train_y_data_03))
    for i in range(len(train_y_data)):
        train_y_data_onehot.append(RetrunOneHot(train_y_data[i]))
    for i in range(len(train_y_data_03)):
        train_y_data_03_onehot.append(RetrunOneHot(train_y_data_03[i]))

    maxBatch = int(len(train_x_data_03_bias) / batch)

    m1 = Model(batch)
    m2 = Model(batch)

    print("batch size   = ", batch)
    print("batch Number = ", maxBatch)
    count = 0
    for i in range(epoch):  # 10 번 반복
        Eavg = 0.
        Eavg2 = 0.
        startNumber = 0
        for j in range(maxBatch):  # 200번 반복
            x_batch = train_x_data_bias[startNumber:startNumber + 100]
            y_batch = train_y_data_onehot[startNumber:startNumber + 100]
            x_batch2 = train_x_data_03_bias[startNumber:startNumber + 100]
            y_batch2 = train_y_data_03_onehot[startNumber:startNumber + 100]
            if (len(x_batch) != 0):
                Eav, pred = m1.forward(x_batch, y_batch)
                m1.backward(pred, y_batch, x_batch)

                Eav2, pred2 = m2.forward(x_batch2, y_batch2)
                m2.backward(pred2, y_batch2, x_batch2)

                Eavg = Eavg + Eav
                Eavg2 = Eavg2 + Eav2
                startNumber = startNumber + 100
        print("Epoch ", i + 1, "Eavg : ", Eavg/maxBatch)
        print("Epoch ", i + 1, "Eavg2 : ", Eavg2 / maxBatch)

    test = np.loadtxt('TestDataset.csv', delimiter=',', dtype=np.float32)
    test_x_data = test[:, 0:-1]
    test_bias = [[1.] * 1 for i in range(len(test_x_data))]
    test_x_data_bias = np.concatenate((test_bias, test_x_data), axis=1)
    test_y_data = test[:, [-1]]
    test_y_data_onehot = []
    for i in range(len(test_y_data)):
        test_y_data_onehot.append(RetrunOneHot(test_y_data[i]))

    test = np.loadtxt('TestDataset_03.csv', delimiter=',', dtype=np.float32)
    test_x_data_03 = test[:, 0:-1]
    test_x_data_03_bias = np.concatenate((test_bias, test_x_data_03), axis=1)
    test_y_data_03 = test[:, [-1]]
    test_y_data_03_onehot = []

    for i in range(len(test_y_data_03)):
        test_y_data_03_onehot.append(RetrunOneHot(test_y_data_03[i]))

    startNumber = 0
    Aavg = 0.
    Aavg2 = 0.
    a = 0
    for i in range(len(test_x_data_bias)): # 200번 반복
        x_batch = test_x_data_bias[startNumber:startNumber+batch]
        y_batch = test_y_data_onehot[startNumber:startNumber+batch]
        x_batch2 = test_x_data_03_bias[startNumber:startNumber+batch]
        y_batch2 = test_y_data_03_onehot[startNumber:startNumber+batch]
        if(len(x_batch)!= 0):
            accuracy, pred = m1.Accuracy(x_batch, y_batch, batch)
            accuracy2, pred2 = m2.Accuracy(x_batch2, y_batch2, batch)
            if(a == 0):
                print("Label : ", np.argmax(y_batch, axis=1))
                print("pred : ", np.argmax(pred, axis=1))
                a += 1
            Aavg = Aavg + accuracy
            Aavg2 = Aavg2 + accuracy2
            startNumber = startNumber + 100

    print("Aavg : {}%".format((float(Aavg)/len(test_x_data_bias))*100))

    print("Aavg2 : {}%".format((float(Aavg2) / len(test_x_data_03_bias))*100))