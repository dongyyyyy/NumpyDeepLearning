import numpy as np
from math import exp
from DeepLearningFunction import *
learning_rate = 0.01

def V_Sigmoid():
    sigmoid = np.vectorize(Sigmoid)
    return sigmoid

def Sigmoid(x):
    return 1/(1 + exp(-x))
def ReLU(x):
    return max(0,x)

if __name__ == "__main__":

    batch = 100
    epoch = 10
    startNumber = 0

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

        z4 = np.dot(h3,W4)
        h4 = v_sigmoid(z4)

        z5 = np.dot(h4,W5)
        o = v_sigmoid(z5)
        print("o shape : ", o.shape)
        e = np.sum(np.square(y - o))/2
        '''
        print(o.shape)
        print("="*20)
        print(y)
        print("=" * 20)
        print("o : ",o)
        print("=" * 20)
        print("y : ",y)
        print("=" * 20)
        print("y-o : ",y-o)
        print("=" * 20)
        print("e: ",e)
        '''
        return e , o,h1,h2,h3,h4,z1,z2,z3,z4,z5

    def backward(x,y,W1,W2,W3,W4,W5,input_data,h1,h2,h3,h4):
        sig5 = x*(1-x)
        local_param5 = (y-x)*sig5 # 국부적 기울기 e(n)*f`(x) = e(n)*o(n)[1-o(n)] 출력노드의 국부적 기울기
        #local_param5 = 각 출력 노드에서의 지역 기울기 8
        #h4 = 각 노드의 출력 값 10
        print(local_param5.shape)
        print(h4.shape)
        result = HandFunction(h4,local_param5,batch)
        delta_o = (-learning_rate)*result   # h4 = h(l-1) , w5 = f'(z(n)) gl(n)
        W5 = W5 + delta_o # 새로운 W5값 W(n+1) = W(n) + delta_W(n)

        local_param4 = (h4)*(1-h4)
        result = HandFunction(h3,local_param4,batch)
        delta_h4 = (-learning_rate)*result
        W4 = W4 + delta_h4

        local_param3 = (h3)*(1-h3)
        result = HandFunction(h2,local_param3,batch)
        delta_h3 = (-learning_rate)*result
        W3 = W3 + delta_h3

        local_param2 = (h2)*(1-h2)
        result = HandFunction(h1,local_param2,batch)
        delta_h2 = (-learning_rate)*result
        W2 = W2 + delta_h2

        local_param1 = (h1)*(1-h1)
        result = HandFunction(input_data,local_param1,batch)
        delta_h1 = (-learning_rate)*result
        W1 = W1 + delta_h1

        return W1,W2,W3,W4,W5

    maxBatch = int(len(train_x_data)/batch)
    print("batch size   = ", batch)
    print("batch Number = ", maxBatch)

    for i in range(epoch): # 10 번 반복
        for j in range(maxBatch): # 200번 반복
            x_batch = train_x_data[startNumber:startNumber+100]
            y_batch = train_y_data_onehot[startNumber:startNumber+100]
            Eav, error, h1,h2,h3,h4,z1,z2,z3,z4,z5 = forward(x_batch,y_batch)
            print("Eav : ",Eav)
            W1,W2,W3,W4,W5 =backward(error,y_batch,W1,W2,W3,W4,W5,x_batch,h1,h2,h3,h4)
            #print(W1)
            startNumber = startNumber + 100
