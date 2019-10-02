import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

learning_rate1 = 0.1
learning_rate2 = 0.01
learning_rate3 = 0.001

point_class = 8


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
    Y = tf.placeholder(tf.float32, [None, 8])# 0 ~ 7

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

    #print(train_y_data_onehot_03)
    print("데이터 총 개수 : ", len(train_y_data_03))

    maxBatch = int(len(train_x_data) / batch)

    print("batch size   = ", batch)
    print("batch Number = ", maxBatch)
    count = 0
    sess = tf.Session()
    train_y_data_onehot = tf.reshape(tf.one_hot(train_y_data,point_class),[-1,point_class]).eval(session=sess)
    train_y_data_03_onehot = tf.reshape(tf.one_hot(train_y_data_03,point_class),[-1,point_class]).eval(session=sess)
    m1_1 = Model_Adam(sess, "m1_1", learning_rate1)
    m2_1 = Model_Grad(sess, "m2_1", learning_rate1)
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
    writer = tf.summary.FileWriter('Logs/classification.log')
    writer.add_graph(sess.graph)
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
                cost_val, _ = m1_1.train(x_batch2, y_batch2)
                cost_val2, _ = m2_1.train(x_batch2, y_batch2)
                cost_val_2, _ = m1_2.train(x_batch2, y_batch2)
                cost_val2_2, _ = m2_2.train(x_batch2, y_batch2)
                cost_val_3, _ = m1_3.train(x_batch2, y_batch2)
                cost_val2_3, _ = m2_3.train(x_batch2, y_batch2)
                Eavg += cost_val
                Eavg2 += cost_val2
                Eavg_2 += cost_val_2
                Eavg2_2 += cost_val2_2
                Eavg_3 += cost_val_3
                Eavg2_3 += cost_val2_3
                startNumber = startNumber + 100
            else:
              break
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

    test = np.loadtxt('TestDataset_03.csv', delimiter=',', dtype=np.float32)
    test_x_data_03 = test[:, 0:-1]
    test_y_data_03 = test[:, [-1]]
    test_y_data_onehot = tf.reshape(tf.one_hot(test_y_data_03, point_class), [-1, point_class]).eval(session=sess)
    startNumber = 0
    Aavg = 0.
    Aavg2 = 0.
    Aavg_2 = 0.
    Aavg2_2 = 0.
    Aavg_3 = 0.
    Aavg2_3 = 0.
    a = 0
    print("batch: ", batch)
    for i in range(len(test_x_data_03)):  # 200번 반복
        x_batch2 = test_x_data_03[startNumber:startNumber + batch]
        y_batch2 = test_y_data_onehot[startNumber:startNumber + batch]
        if (len(x_batch2) != 0):
            accuracy = m1_1.get_accuracy(x_batch2, y_batch2)
            accuracy2 = m2_1.get_accuracy(x_batch2, y_batch2)
            accuracy_2 = m1_2.get_accuracy(x_batch2, y_batch2)
            accuracy2_2 = m2_2.get_accuracy(x_batch2, y_batch2)
            accuracy_3 = m1_3.get_accuracy(x_batch2, y_batch2)
            accuracy2_3 = m2_3.get_accuracy(x_batch2, y_batch2)
            Aavg = Aavg + accuracy
            Aavg2 = Aavg2 + accuracy2
            Aavg_2 += accuracy_2
            Aavg2_2 += accuracy2_2
            Aavg_3 += accuracy_3
            Aavg2_3 += accuracy2_3
            print(accuracy)
            startNumber = startNumber + 100
        else:
            break

    print("Adam 0.1 : {}%".format((float(Aavg) / len(test_x_data_03)) * 100))
    print("Grad 0.1 : {:.4f}%".format((float(Aavg2) / len(test_x_data_03)) * 100))
    print("Adam 0.01 : {}%".format((float(Aavg_2) / len(test_x_data_03)) * 100))
    print("Grad 0.01 : {:.4f}%".format((float(Aavg2_2) / len(test_x_data_03)) * 100))
    print("Adam 0.001 : {}%".format((float(Aavg_3) / len(test_x_data_03)) * 100))
    print("Grad 0.001 : {:.4f}%".format((float(Aavg2_3) / len(test_x_data_03)) * 100))
