# coding:UTF-8

import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="D:/GitHub/tensorflow_tutorial/data/mnist.npz")

def array_2_one_hot(arr,n=10):
    size= arr.shape[0]
    res = np.zeros([size,n])
    for i in range(size):
        res[i][arr[i]]=1
    return res


x_train = np.reshape(x_train.astype(np.float32), [-1,28,28,1])
x_test = np.reshape(x_test.astype(np.float32), [-1,28,28,1])
y_train_one_hot = array_2_one_hot(y_train)
y_test_one_hot = array_2_one_hot(y_test)


with tf.name_scope("mnistCnn"):
    # 定义占位符，变量W，b
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])

    kernel_1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))#这里的1, 32的意思是输入一个输出32个
    bias_1 = tf.Variable(tf.zeros([32]))
    conv_1 = tf.nn.relu(tf.nn.conv2d(input=x,filter=kernel_1,strides=[1,1,1,1],padding='SAME') + bias_1)
    layer_1 = pool_1=tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    kernel_2 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1))
    bias_2 = tf.Variable(tf.zeros([64]))
    conv_2 = tf.nn.relu(tf.nn.conv2d(input=layer_1,filter=kernel_2,strides=[1,1,1,1],padding='SAME')+ bias_2)
    layer_2 = pool_2 = tf.nn.max_pool(conv_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W_3 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
    bias_3 = tf.Variable(tf.zeros([1024]))
    flatten = tf.reshape(layer_2,[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(flatten,W_3)+bias_3)
    keep_prob = tf.placeholder(tf.float32)
    layer_3 = dropout = tf.nn.dropout(fc,keep_prob=keep_prob)

    W_4 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
    bias_4 = tf.Variable(tf.zeros([10]))
    layer_4 = tf.matmul(layer_3,W_4)+bias_4
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_4, labels=y))

    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy) # 之前设为0.01，陷入
    accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(layer_4,1),tf.argmax(y,1)),tf.float32))

epoch = 1000
batch_size = 64
data_size = x_train.shape[0]
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(epoch+1):
            for i in range(data_size//batch_size):
                sess.run(train_step, feed_dict={x:x_train[i*batch_size:(i+1)*batch_size], y:y_train_one_hot[i*batch_size:(i+1)*batch_size], keep_prob:0.5})
            if _%20 == 0: # 60轮，99%
                acc = sess.run(accuracy,feed_dict={x:x_test, y:y_test_one_hot, keep_prob:1.0})
                print("第",_,"轮，准确率：",acc)

