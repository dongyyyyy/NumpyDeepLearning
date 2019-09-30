import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from DeepLearningFunction import *

learning_rate1 = 0.1
learning_rate2 = 0.01
learning_rate3 = 0.001


class Model_Grad:
    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3])
            self.Y = tf.placeholder(tf.float32, [None, 8])

            #W1 = tf.Variable(tf.random_normal([3, 24]))
            #W1 = tf.get_variable("W1", shape=[3,24], initializer=tf.contrib.layers.xavier_initializer())
            #W1 = tf.get_variable("W1", shape=[3, 24], initializer=tf.compat.v1.initializers.he_normal())
            '''
            tf.constant_initializer(value) 제공된 값으로 모든 것을 초기화합니다,
            tf.random_uniform_initializer(a, b) [a, b]를 균일하게 초기화 합니다,
            tf.random_normal_initializer(mean, stddev) 주어진 평균 및 표준 편차로 정규 분포에서 초기화합니다.
            '''
            b1 = tf.Variable(tf.random_normal([24]))

            W2 = tf.Variable(tf.random_normal([24, 8]))
            b2 = tf.Variable(tf.random_normal([8]))

            h1 = tf.matmul(self.X, W1) + b1

            self.logits = tf.matmul(h1, W2) + b2

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data})


class Model_Adam:
    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3])  # 데이터를 저장하는 일종의 통
            self.Y = tf.placeholder(tf.float32, [None, 8])
            '''
            tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
            정규분포로부터의 난수값을 반환합니다
            shape: 정수값의 1-D 텐서 또는 파이썬 배열. 반환값 텐서의 shape입니다.
            mean: 0-D 텐서 또는 dtype타입의 파이썬 값. 정규분포의 평균값.
            stddev: 0-D 텐서 또는 dtype타입의 파이썬 값. 정규분포의 표준 편차.
            dtype: 반환값의 타입.
            seed: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 set_random_seed를 보십시오.
            name: 연산의 명칭 (선택사항).
            반환값 : 정규 난수값들로 채워진 shape으로 정해진 텐서

            np.random.normal(0, 0.3, 3)
            mean : 0
            stddev : 0.3
            shape : 3
            와 같은 형태

            tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
            균등분포로부터의 난수값을 반환합니다
            shape: 정수값의 D-1 텐서 또는 파이썬 배열. 반환값 텐서의 shape입니다.
            minval: 0-D 텐서 또는 dtype타입의 파이썬 값. 난수값 생성 구간의 하한입니다. 기본값은 0입니다.
            maxval: 0-D 텐서 또는 dtype타입의 파이썬 값. 난수값 생성 구간의 상한입니다. dtype이 실수형일 경우 기본값은 1입니다.
            dtype: 반환값의 타입: float32, float64, int32, 또는 int64.
            seed: 파이썬 정수. 분포의 난수 시드값을 생성하는데에 사용됩니다. 동작 방식은 set_random_seed를 보십시오.
            name: 연산의 명칭 (선택사항).
            반환값 : 균등 난수값들로 채워진 shape으로 정해진 텐서.
            '''
            W1 = tf.Variable(tf.random_normal([3, 24]))
            b1 = tf.Variable(tf.random_normal([24]))

            W2 = tf.Variable(tf.random_normal([24, 8]))
            b2 = tf.Variable(tf.random_normal([8]))

            h1 = tf.matmul(self.X, W1) + b1

            self.logits = tf.matmul(h1, W2) + b2

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data})


if __name__ == "__main__":
    X = tf.placeholder(tf.float32, [None, 3])
    Y = tf.placeholder(tf.float32, [None, 8])

    W1 = tf.Variable(tf.random_normal([3, 24]))
    b1 = tf.Variable(tf.random_normal([24]))

    W2 = tf.Variable(tf.random_normal([24, 8]))
    b2 = tf.Variable(tf.random_normal([8]))

    batch = 100
    epoch = 10
    startNumber = 0

    data = np.loadtxt('TrainDataset.csv', delimiter=',', dtype=np.float32)
    train_x_data = data[:, 0:-1]
    train_y_data = data[:, [-1]]
    train_y_data_onehot = []

    data = np.loadtxt('TrainDataset_03.csv', delimiter=',', dtype=np.float32)
    train_x_data_03 = data[:, 0:-1]
    train_y_data_03 = data[:, [-1]]
    train_y_data_03_onehot = []

    print("데이터 총 개수 : ", len(train_y_data_03))
    for i in range(len(train_y_data)):
        train_y_data_onehot.append(RetrunOneHot(train_y_data[i]))
    for i in range(len(train_y_data_03)):
        train_y_data_03_onehot.append(RetrunOneHot(train_y_data_03[i]))

    maxBatch = int(len(train_x_data) / batch)

    print("batch size   = ", batch)
    print("batch Number = ", maxBatch)
    count = 0
    sess = tf.Session()

    m1 = Model_Adam(sess, "m1_1", learning_rate1)
    m2 = Model_Grad(sess, "m2_1", learning_rate1)
    m1_2 = Model_Adam(sess, "m1_2", learning_rate2)
    m2_2 = Model_Grad(sess, "m2_2", learning_rate2)
    m1_3 = Model_Adam(sess, "m1_3", learning_rate3)
    m2_3 = Model_Grad(sess, "m2_3", learning_rate3)

    x1_1 = []
    x2_1 = []
    x1_2 = []
    x2_2 = []
    x1_3 = []
    x2_3 = []

    sess.run(tf.global_variables_initializer())

    for i in range(epoch):  # 10 번 반복
        Eavg = 0.
        Eavg2 = 0.
        Eavg_2 = 0.
        Eavg2_2 = 0.
        Eavg_3 = 0.
        Eavg2_3 = 0.
        startNumber = 0
        for j in range(maxBatch):  # 200번 반복
            x_batch = train_x_data[startNumber:startNumber + 100]
            y_batch = train_y_data_onehot[startNumber:startNumber + 100]
            x_batch2 = train_x_data_03[startNumber:startNumber + 100]
            y_batch2 = train_y_data_03_onehot[startNumber:startNumber + 100]
            if (len(x_batch) != 0):
                cost_val, _ = m1.train(x_batch, y_batch)
                cost_val2, _ = m2.train(x_batch, y_batch)
                cost_val_2, _ = m1_2.train(x_batch, y_batch)
                cost_val2_2, _ = m2_2.train(x_batch, y_batch)
                cost_val_3, _ = m1_3.train(x_batch, y_batch)
                cost_val2_3, _ = m2_3.train(x_batch, y_batch)
                Eavg += cost_val
                Eavg2 += cost_val2
                Eavg_2 += cost_val_2
                Eavg2_2 += cost_val2_2
                Eavg_3 += cost_val_3
                Eavg2_3 += cost_val2_3
                startNumber = startNumber + 100
        print("Epoch ", i + 1, "Eavg_Adam_1 : ", Eavg / maxBatch)
        x1_1.append(Eavg / maxBatch)
        print("Epoch ", i + 1, "Eavg_Grad_1 : ", Eavg2 / maxBatch)
        x2_1.append(Eavg2 / maxBatch)
        print("Epoch ", i + 1, "Eavg_Adam_2 : ", Eavg_2 / maxBatch)
        x1_2.append(Eavg_2 / maxBatch)
        print("Epoch ", i + 1, "Eavg_Grad_2 : ", Eavg2_2 / maxBatch)
        x2_2.append(Eavg2_2 / maxBatch)
        print("Epoch ", i + 1, "Eavg_Adam_2 : ", Eavg_3 / maxBatch)
        x1_3.append(Eavg_3 / maxBatch)
        print("Epoch ", i + 1, "Eavg_Grad_2 : ", Eavg2_3 / maxBatch)
        x2_3.append(Eavg2_3 / maxBatch)

    plt.xlabel('Epoch')
    plt.ylabel('Cost')

    plt.plot(x1_1, label='Adam_0.1')
    plt.plot(x2_1, label='Grad_0.1')
    plt.plot(x1_2, label='Adam_0.01')
    plt.plot(x2_2, label='Grad_0.01')
    plt.plot(x1_3, label='Adam_0.001')
    plt.plot(x2_3, label='Grad_0.001')

    plt.legend(fontsize='x-large')
    plt.show()