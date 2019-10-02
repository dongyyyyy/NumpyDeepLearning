import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

learning_rate = 0.001

point_class = 8

indices = [0, 1, 2, 3, 4, 5, 6, 7]

print(tf.one_hot(indices=indices, depth=8))


class Model_normal:
    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3])
            self.Y = tf.placeholder(tf.float32, [None, 8])

            W1 = tf.Variable(tf.random_normal([3, 6]))
            b1 = tf.Variable(tf.random_normal([6]))

            W2 = tf.Variable(tf.random_normal([6, 8]))
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


class Model_xavier:
    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3])
            self.Y = tf.placeholder(tf.float32, [None, 8])

            W1 = tf.get_variable("W1_2", shape=[3, 6], initializer=tf.contrib.layers.xavier_initializer())  # xavier초기화
            b1 = tf.Variable(tf.random_normal([6]))

            W2 = tf.get_variable("W2_2", shape=[6, 8], initializer=tf.contrib.layers.xavier_initializer())  # xavier초기화
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


class Model_he_norm:
    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 3])
            self.Y = tf.placeholder(tf.float32, [None, 8])

            W1 = tf.get_variable("W1_3", shape=[3, 6], initializer=tf.compat.v1.initializers.he_normal())  # he_정규분포 초기화
            b1 = tf.Variable(tf.random_normal([6]))

            W2 = tf.get_variable("W2_3", shape=[6, 8], initializer=tf.contrib.layers.xavier_initializer())  # xavier초기화
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

    batch = 100
    epoch = 10
    startNumber = 0

    data = np.loadtxt('TrainDataset_03.csv', delimiter=',', dtype=np.float32)
    train_x_data_03 = data[:, 0:-1]
    train_y_data_03 = data[:, -1]

    # print(train_y_data_onehot_03)
    print("데이터 총 개수 : ", len(train_y_data_03))

    maxBatch = int(len(train_x_data_03) / batch)

    print("batch size   = ", batch)
    print("batch Number = ", maxBatch)
    count = 0
    sess1 = tf.Session()
    sess2 = tf.Session()
    sess3 = tf.Session()

    train_y_data_03_onehot = tf.reshape(tf.one_hot(train_y_data_03, point_class), [-1, point_class]).eval(session=sess1)
    m_norm = Model_normal(sess1, "m_norm", learning_rate)
    m_xavier = Model_xavier(sess2, "m_xavier", learning_rate)
    m_he_norm = Model_he_norm(sess3, "m_he_norm", learning_rate)

    x_1 = []
    x_2 = []
    x_3 = []

    sess1.run(tf.global_variables_initializer())
    sess2.run(tf.global_variables_initializer())
    sess3.run(tf.global_variables_initializer())

    for i in range(epoch):  # 10 번 반복
        Eavg1 = 0.
        Eavg2 = 0.
        Eavg3 = 0.
        startNumber = 0
        for j in range(maxBatch):  # 200번 반복
            x_batch2 = train_x_data_03[startNumber:startNumber + 100]
            y_batch2 = train_y_data_03_onehot[startNumber:startNumber + 100]
            if (len(x_batch2) != 0):
                cost_val1, _ = m_norm.train(x_batch2, y_batch2)
                cost_val2, _ = m_xavier.train(x_batch2, y_batch2)
                cost_val3, _ = m_he_norm.train(x_batch2, y_batch2)

                Eavg1 += cost_val1
                Eavg2 += cost_val2
                Eavg3 += cost_val3

                startNumber = startNumber + 100
            else:
                break
        print("Epoch ", i + 1, "Random_normal : ", Eavg1 / maxBatch)
        x_1.append(Eavg1 / maxBatch)
        print("Epoch ", i + 1, "Xavier : ", Eavg2 / maxBatch)
        x_2.append(Eavg2 / maxBatch)
        print("Epoch ", i + 1, "He_norm : ", Eavg3 / maxBatch)
        x_3.append(Eavg3 / maxBatch)

