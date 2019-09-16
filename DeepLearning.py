import numpy as np
from math import exp
from DeepLearningFunction import *
learning_rate = 0.00001

def V_Sigmoid():
    sigmoid = np.vectorize(Sigmoid)
    return sigmoid

def Sigmoid(x):
    try:
        return 1 / (1 + exp(-x))
    except OverflowError:
        return float('inf')

def ReLU(x):
    return max(0,x)

if __name__ == "__main__":

    batch = 100
    epoch = 1
    startNumber = 0

    data = np.loadtxt('TrainDataset.csv',delimiter=',',dtype=np.float32)
    train_x_data = data[:,0:-1]
    bias = [[1.] * 1 for i in range(len(train_x_data))]
    train_x_data_bias = np.concatenate((bias,train_x_data), axis=1)
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


    W1 = np.random.rand(4,12) # X = 3 / L1의 노드의 수 = 10
    W2 = np.random.rand(12, 24)  # X = 3 / L1의 노드의 수 = 10
    W3 = np.random.rand(24, 48)  # X = 3 / L1의 노드의 수 = 10
    W4 = np.random.rand(48, 96)  # X = 3 / L1의 노드의 수 = 10
    W5 = np.random.rand(96, 192)  # X = 3 / L1의 노드의 수 = 10
    W6 = np.random.rand(192,192)
    W7 = np.random.rand(192,96)
    W8 = np.random.rand(96,8)
    #print(W1)
    v_sigmoid = V_Sigmoid()
    h1 = np.array(np.zeros(12))
    h2 = np.array(np.zeros(24))
    h3 = np.array(np.zeros(48))
    h4 = np.array(np.zeros(96))
    h5 = np.array(np.zeros(192))
    h6 = np.array(np.zeros(192))
    h7 = np.array(np.zeros(96))
    def forward(x,y):
        z1 = np.dot(x,W1)
        #print("z1:",z1)
        h1 = v_sigmoid(z1)
        h1 = MakeFristOne(h1)
        #print("h1 :", h1)
        z2 = np.dot(h1,W2)
        h2 = v_sigmoid(z2)
        h2 = MakeFristOne(h2)
        #print("h2 :", h2)

        z3 = np.dot(h2,W3)
        h3 = v_sigmoid(z3)
        h3 = MakeFristOne(h3)
        #print("h3 :", h3)

        z4 = np.dot(h3,W4)
        h4 = v_sigmoid(z4)
        h4 = MakeFristOne(h4)
        #print("h4 :", h3)

        z5 = np.dot(h4,W5)
        h5 = v_sigmoid(z5)
        h5 = MakeFristOne(h5)
        #print("h5 :", h5)

        z6 = np.dot(h5,W6)
        h6 = v_sigmoid(z6)
        h6 = MakeFristOne(h6)
        #print("h6 :", h6)
        z7 = np.dot(h6, W7)
        h7 = v_sigmoid(z7)
        h7 = MakeFristOne(h7)

        #print("h7 :",h7)
        z8 = np.dot(h7,W8)
        #print("z8:",z8)
        #print("z8 : ",z8)
        o = ODivideFunction(np.exp(z8), np.sum(np.exp(z8),axis=1))
        #print("x: ",x)
        #print("o:",o)
        #print("o : ",o)
        #e = np.sum(np.square(y - o))/2/batch # 해당 부분 수정 필요
        e = np.mean(-np.sum(y*np.log(o),axis=1))
        return e , o,h1,h2,h3,h4,h5,h6,h7,z1,z2,z3,z4,z5,z6,z7,z8

    def Accuracy(x,y,batch):
        z1 = np.dot(x,W1)
        #print("z1:",z1)
        h1 = v_sigmoid(z1)
        h1 = MakeFristOne(h1)
        #print("h1 :", h1)
        z2 = np.dot(h1,W2)
        h2 = v_sigmoid(z2)
        h2 = MakeFristOne(h2)
        #print("h2 :", h2)

        z3 = np.dot(h2,W3)
        h3 = v_sigmoid(z3)
        h3 = MakeFristOne(h3)
        #print("h3 :", h3)

        z4 = np.dot(h3,W4)
        h4 = v_sigmoid(z4)
        h4 = MakeFristOne(h4)
        #print("h4 :", h3)

        z5 = np.dot(h4,W5)
        h5 = v_sigmoid(z5)
        h5 = MakeFristOne(h5)
        #print("h5 :", h5)

        z6 = np.dot(h5,W6)
        h6 = v_sigmoid(z6)
        h6 = MakeFristOne(h6)
        #print("h6 :", h6)
        z7 = np.dot(h6, W7)
        h7 = v_sigmoid(z7)
        h7 = MakeFristOne(h7)

        #print("h7 :",h7)
        z8 = np.dot(h7,W8)
        #print("z8:",z8)
        #print("z8 : ",z8)
        o = ODivideFunction(np.exp(z8), np.sum(np.exp(z8),axis=1))
        print("o.shape : ",o.shape)
        accuracy = 0.
        for i in range(batch):
            print("Accuruacy : ",o[i],y[i])
            if (np.argmax(o[i])==np.argmax(y[i])):
                accuracy = accuracy + 1.
        return accuracy/batch


    def backward(x,y,W1,W2,W3,W4,W5,W6,W7,W8,input_data,h1,h2,h3,h4,h5,h6,h7):
        local_param8 = (x-y)
        #sig8 = x*(1-x)
        #local_param8 = (y-x)*sig8 # 출력노드 국부적기울기
        result = HandFunction(h7,local_param8,batch)
        delta_o = (-learning_rate)*result
        NW8 = W8 + delta_o

        sig7 = (h7) * (1 - h7)
        local_param7 = HandFunction2(np.sum(local_param8,axis=1),sig7,batch) #  (100,8) , (100, 96)
        local_param7 = HandFunction1(np.sum(W8,axis=1),local_param7,batch) # (96 , 1) , ( 100, 96 ) -> ( 100, 96 )
        result = HandFunction(h6, local_param7, batch) # h6 = (100,192) local_param7 = (100,96)
        delta_h7 = (-learning_rate) * result
        NW7 = W7 + delta_h7

        # (192,96)
        sig6 = (h6)*(1-h6)
        local_param6 = HandFunction2(np.sum(local_param7, axis=1), sig6, batch) # (100, 96) , (100,192) -> (192, 96)
        local_param6 = HandFunction1(np.sum(W7, axis=1), local_param6, batch) #
        result = HandFunction(h5,local_param6,batch) # 100,192 / 100 , 96
        delta_h6 = (-learning_rate)*result   # h4 = h(l-1) , w5 = f'(z(n)) gl(n)
        NW6 = W6 + delta_h6 # 새로운 W5값 W(n+1) = W(n) + delta_W(n)

        sig5 = (h5) * (1 - h5)
        local_param5 = HandFunction2(np.sum(local_param6, axis=1), sig5, batch)
        local_param5 = HandFunction1(np.sum(W6, axis=1), local_param5, batch)
        result = HandFunction(h4, local_param5, batch)
        delta_h5 = (-learning_rate) * result
        NW5 = W5 + delta_h5

        sig4 = (h4) * (1 - h4)
        local_param4 = HandFunction2(np.sum(local_param5, axis=1), sig4, batch)
        local_param4 = HandFunction1(np.sum(W5, axis=1), local_param4, batch)
        result = HandFunction(h3, local_param4, batch)
        delta_h4 = (-learning_rate) * result
        NW4 = W4 + delta_h4

        sig3 = (h3) * (1 - h3)
        local_param3 = HandFunction2(np.sum(local_param4, axis=1), sig3, batch)
        local_param3 = HandFunction1(np.sum(W4, axis=1), local_param3, batch)
        result = HandFunction(h2, local_param3, batch)
        delta_h3 = (-learning_rate) * result
        NW3 = W3 + delta_h3

        sig2 = (h2) * (1 - h2)
        local_param2 = HandFunction2(np.sum(local_param3, axis=1), sig2, batch)
        local_param2 = HandFunction1(np.sum(W3, axis=1), local_param2, batch)
        result = HandFunction(h1, local_param2, batch)
        delta_h2 = (-learning_rate) * result
        NW2 = W2 + delta_h2

        sig1 = (h1) * (1 - h1)
        local_param1 = HandFunction2(np.sum(local_param2, axis=1), sig1, batch)
        local_param1 = HandFunction1(np.sum(W2, axis=1), local_param1, batch)
        result = HandFunction(input_data, local_param1, batch)
        delta_h1 = (-learning_rate) * result
        NW1 = W1 + delta_h1
        return NW1,NW2,NW3,NW4,NW5,NW6,NW7,NW8

    maxBatch = int(len(train_x_data_bias)/batch)
    count = 0
    for i in range(epoch): # 10 번 반복
        Eavg = 0.
        startNumber = 0
        for j in range(maxBatch): # 200번 반복
            x_batch = train_x_data_bias[startNumber:startNumber+batch]
            y_batch = train_y_data_onehot[startNumber:startNumber+batch]
            if(len(x_batch)!= 0):
                Eav, error, h1,h2,h3,h4,h5,h6,h7,z1,z2,z3,z4,z5,z6,z7,z8 = forward(x_batch,y_batch)
                Eavg = Eavg + Eav
                W1,W2,W3,W4,W5,W6,W7,W8 = backward(error,y_batch,W1,W2,W3,W4,W5,W6,W7,W8,x_batch,h1,h2,h3,h4,h5,h6,h7)
                #print("batch : ",j+1,"Eav : ",Eav)
                #print(W1)
                startNumber = startNumber + 100
        print("epoch : ",i+1,"Eavg : ",Eavg/maxBatch)


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
            #print("batch : ",j+1,"Eav : ",Eav)
            #print(W1)
            #print(accuracy)
            Aavg = Aavg + accuracy
            startNumber = startNumber + 100
    print("Aavg : {:.2f}%".format((Aavg/len(test_x_data_bias))*100))


