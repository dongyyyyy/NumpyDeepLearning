import tensorflow as tf
import matplotlib.pyplot as plt

W1 = tf.Variable(tf.random_normal([1000])) # 정규 분포
W2 = tf.get_variable("W2", shape=[1000], initializer=tf.contrib.layers.xavier_initializer()) # xavier초기화
W3 = tf.get_variable("W3", shape=[1000], initializer=tf.compat.v1.initializers.he_normal()) # he_정규분포 초기화
tf.Variable()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

W1_value = sess.run(W1)
W2_value = sess.run(W2)
W3_value = sess.run(W3)


plt.title('Normal distribution')
plt.xlabel('value')
plt.ylabel('count')
plt.hist(W1_value)
plt.show()

plt.title('xavier')
plt.xlabel('value')
plt.ylabel('count')
plt.hist(W2_value)
plt.show()



plt.title('He_normal')
plt.xlabel('value')
plt.ylabel('count')
plt.hist(W3_value)
plt.show()



#https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer
#https://www.tensorflow.org/api_docs/python/tf/initializers/he_normal