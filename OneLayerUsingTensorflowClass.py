import numpy as np
from DeepLearningFunction import *
import matplotlib.pyplot as plt
import tensorflow as tf

learning_rate = 0.01

class Model_Grad:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3])
            self.Y = tf.placeholder(tf.float32, [None, 8])

            W1 = tf.Variable(tf.random_normal([3, 24]))
            b1 = tf.Variable(tf.random_normal([24]))

            W2 = tf.Variable(tf.random_normal([24, 8]))
            b2 = tf.Variable(tf.random_normal([8]))

            h1 = tf.matmul(self.X, W1) + b1

            self.logits = tf.matmul(h1, W2) + b2

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

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
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3]) # 데이터를 저장하는 일종의 통
            self.Y = tf.placeholder(tf.float32, [None, 8])

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
            learning_rate=learning_rate).minimize(self.cost)

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


point_class = 8

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


    data = np.loadtxt('TrainDataset_03.csv', delimiter=',', dtype=np.float32)
    train_x_data_03 = data[:, 0:-1]
    train_y_data_03 = data[:, [-1]]

    print("데이터 총 개수 : ", len(train_y_data_03))

    maxBatch = int(len(train_x_data) / batch)

    print("batch size   = ", batch)
    print("batch Number = ", maxBatch)
    count = 0
    sess = tf.Session()
    sess1 = tf.Session()

    m1 = Model_Adam(sess, "m1")
    m2 = Model_Grad(sess1, "m2")

    sess.run(tf.global_variables_initializer())
    train_y_data_onehot = tf.reshape(tf.one_hot(train_y_data, depth=point_class), [-1, point_class]).eval(session=sess)

    sess1.run(tf.global_variables_initializer())

    train_y_data_03_onehot = tf.reshape(tf.one_hot(train_y_data_03, point_class), [-1, point_class]).eval(session=sess1)

    for i in range(epoch):  # 10 번 반복
        Eavg = 0.
        Eavg2 = 0.
        startNumber = 0
        for j in range(maxBatch):  # 200번 반복
            x_batch = train_x_data[startNumber:startNumber + 100]
            y_batch = train_y_data_onehot[startNumber:startNumber + 100]
            x_batch2 = train_x_data_03[startNumber:startNumber + 100]
            y_batch2 = train_y_data_03_onehot[startNumber:startNumber + 100]
            if (len(x_batch) != 0):
                cost_val, _ = m1.train(x_batch,y_batch)
                cost_val2, _ = m2.train(x_batch,y_batch)
                Eavg += cost_val
                Eavg2 += cost_val2
                startNumber = startNumber + 100
        print("Epoch ", i + 1, "Eavg_Adam : ", Eavg / maxBatch)
        print("Epoch ", i + 1, "Eavg_Grad : ", Eavg2 / maxBatch)
