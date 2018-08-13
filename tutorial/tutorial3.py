# coding:UTF-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with tf.name_scope("r"):
    #数据预生成
    x_data = np.linspace(0, 64, 200)[:,np.newaxis]
    noise = np.random.normal(0, 0.2, x_data.shape)
    y_data = np.square(x_data + 10) + noise

    #定义结构占位置
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    weight_1 = tf.Variable(tf.random_normal([1, 10]))#这里的1, 10的意思是输入一个输出10个
    bias_1 = tf.Variable(tf.zeros([1, 10]))
    wx_add_b_1 = tf.matmul(weight_1, x) + bias_1
    l_1 = tf.nn.tanh(wx_add_b_1)

    weight_2 = tf.Variable(tf.random_normal([10, 1]))#这里的10, 1的意思是输入一个输出10个
    bias_2 = tf.Variable(tf.zeros([1, 1]))
    wx_add_b_2 = tf.matmul(weight_2, x) + bias_2
    prediction = tf.nn.tanh(wx_add_b_2)

    loss = tf.reduce_mean(tf.square(y - prediction))

    #线性训练
    line_train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2000):
            sess.run(line_train, feed_dict={x:x_data, y:y_data})
        prediction_value = sess.run(line_train,feed_dict={x:x_data, y:y_data})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.show()

#with tf.name_scope("mnist_data"):
#    input = tf.variable_scope()