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

    W1 = np.random.rand(4, 12)  # X = 3 / L1의 노드의 수 = 10
    W2 = np.random.rand(12, 24)  # X = 3 / L1의 노드의 수 = 10
    W3 = np.random.rand(24, 12)  # X = 3 / L1의 노드의 수 = 10
    W4 = np.random.rand(12, 8)  # X = 3 / L1의 노드의 수 = 10
    # print(W1)
    v_sigmoid = V_Sigmoid()
    v_relu = V_ReLU()
    h1 = np.array(np.zeros(12))
    h2 = np.array(np.zeros(24))
    h3 = np.array(np.zeros(12))


    def forward(x, y):
        z1 = np.dot(x, W1)
        #print("z1:", z1)
        h1 = v_sigmoid(z1)
        h1 = MakeFristOne(h1)
        #h1 = MakeFristOne(z1)

        z2 = np.dot(h1, W2)
        h2 = v_sigmoid(z2)
        h2 = MakeFristOne(h2)
        #h2 = MakeFristOne(z2)

        z3 = np.dot(h2, W3)
        h3 = v_sigmoid(z3)
        h3 = MakeFristOne(h3)
        #h3 = MakeFristOne(z3)

        z4 = np.dot(h3, W4)
        # print("z8 : ",z8)
        o = ODivideFunction(np.exp(z4), np.sum(np.exp(z4), axis=1))
        #print("y[0:2]: ", np.argmax(y[0]),np.argmax(y[1]))
        #print("o[0:2]:", np.argmax(o[0]),np.argmax(o[1]))
        # print("o : ",o)
        # e = np.sum(np.square(y - o))/2/batch # 해당 부분 수정 필요
        e = np.mean(-np.sum(y * np.log(o), axis=1))
        accuracy = 0.
        for i in range(batch):
            #print("Accuruacy : ",np.argmax(o[i]),np.argmax(y[i]))
            if (np.argmax(o[i])==np.argmax(y[i])):
                accuracy = accuracy + 1.
        #print("Train Accuracy : ",accuracy)

        return e, o, h1, h2, h3, z1, z2, z3, z4


    def backward(x, y, W1, W2, W3, W4, input_data, h1, h2, h3):
        local_param4 = (x - y)
        # sig8 = x*(1-x)
        # local_param8 = (y-x)*sig8 # 출력노드 국부적기울기
        result = HandFunction(h3, local_param4)
        delta_o = (-learning_rate) * result
        NW4 = W4 + delta_o

        sig3 = (h3) * (1 - h3)
        local_param3 = HandFunction2(np.sum(local_param4, axis=1), sig3)  # (100,8) , (100, 96)
        local_param3 = HandFunction1(np.sum(W4, axis=1), local_param3)  # (96 , 1) , ( 100, 96 ) -> ( 100, 96 )
        result = HandFunction(h2, local_param3)  # h6 = (100,192) local_param7 = (100,96)
        delta_h3 = (-learning_rate) * result
        NW3 = W3 + delta_h3

        # (192,96)
        sig2 = (h2) * (1 - h2)
        local_param2 = HandFunction2(np.sum(local_param3, axis=1), sig2)  # (100, 96) , (100,192) -> (192, 96)
        local_param2 = HandFunction1(np.sum(W3, axis=1), local_param2)  #
        result = HandFunction(h1, local_param2)  # 100,192 / 100 , 96
        delta_h2 = (-learning_rate) * result  # h4 = h(l-1) , w5 = f'(z(n)) gl(n)
        NW2 = W2 + delta_h2  # 새로운 W5값 W(n+1) = W(n) + delta_W(n)


        sig1 = (h1) * (1 - h1)
        local_param1 = HandFunction2(np.sum(local_param2, axis=1), sig1)
        local_param1 = HandFunction1(np.sum(W2, axis=1), local_param1)
        result = HandFunction(input_data, local_param1)
        delta_h1 = (-learning_rate) * result
        #print("delta_h1:",delta_h1)
        NW1 = W1 + delta_h1

        return NW1, NW2, NW3, NW4

    def Accuracy(x,y,batch):
        z1 = np.dot(x, W1)
        # print("z1:", z1)
        h1 = v_sigmoid(z1)
        h1 = MakeFristOne(h1)
        # print("h1 :", h1)
        z2 = np.dot(h1, W2)
        h2 = v_sigmoid(z2)
        h2 = MakeFristOne(h2)
        # print("h2 :", h2)

        z3 = np.dot(h2, W3)
        h3 = v_sigmoid(z3)
        h3 = MakeFristOne(h3)
        # print("h3 :", h3)

        z4 = np.dot(h3, W4)
        # print("z4:", z4)
        # print("z8 : ",z8)
        o = ODivideFunction(np.exp(z4), np.sum(np.exp(z4), axis=1))
        #print("o.shape : ",o.shape)
        accuracy = 0.
        for i in range(batch):
            #print("Accuruacy : ",np.argmax(o[i]),np.argmax(y[i]))
            if (np.argmax(o[i])==np.argmax(y[i])):
                accuracy = accuracy + 1.
        #print("batch : ",accuracy/batch)
        return accuracy/batch * 100

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
                Eav, error, h1, h2, h3, z1, z2, z3, z4 = forward(x_batch, y_batch)
                #print("h1 :",h1[0:2],"h2 :",h2[0:2],"h3 :",h3[0:2])
                #print("Batch{}'s Eav : ".format(j+1), Eav)
                W1, W2, W3, W4 = backward(error, y_batch, W1, W2, W3, W4, x_batch, h1,
                                                          h2, h3)
                # print(W1)
                Eavg = Eavg + Eav
                startNumber = startNumber + 100
        print("Epoch ",i+1,"Eavg : ",Eavg/maxBatch)

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
            test_y_data_onehot.append([1,0,0,0,0,0,0,0])
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
    for i in range(len(test_x_data_bias)): # 200번 반복
        x_batch = test_x_data_bias[startNumber:startNumber+batch]
        y_batch = test_y_data_onehot[startNumber:startNumber+batch]
        if(len(x_batch)!= 0):
            accuracy = Accuracy(x_batch,y_batch,batch)
            Aavg = Aavg + accuracy
            startNumber = startNumber + 100
    print("Aavg : {}%".format((Aavg/len(test_x_data_bias))))