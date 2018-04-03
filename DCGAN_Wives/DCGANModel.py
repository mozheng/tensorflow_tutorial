# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 15:03
# @Author  : Mozheng
# @Email   : mozhengweiyi@163.com
# @Site    : 
# @File    : DCGANModel.py
# @Software: PyCharm python3.6
# @Version :
# ---------------------------------------
# This file is focus on :
# 
# ---------------------------------------
import os
import datetime
import math
import numpy as np
import tensorflow as tf
from .DCGANWives import Wives


class DCGANModel:

    def __init__(self):
        """Method for recording a benchmark directly.
        #   GAN模型信息
        #
        """
        self.wives = Wives("../datasets/")    #生成老婆们模型，这是所有万恶之源，下面都是她的属性
        self.image_shape = self.wives.image_shape
        self.each_batch_shape = self.wives.each_batch_shape
        self.each_batch_size = self.wives.each_batch_size
        self.all_batch_sum = self.wives.all_batch_sum

        # 下面数据是我抄的，别问我怎么想的
        self.noise_img_size = 100       # 噪音图片size
        self.gf_size = 64               # 反向卷积输出通道数量（图片生成器）
        self.df_size = 64               # 正向卷积输出通道数量（图像判别器）
        self.epoch_size = 50            # 训练循环次数
        self.learning_rate = 0.0002     # 学习率
        self.beta1 = 0.5                # 优化指数衰减率
        self.sample_size = 64           # 生成图片数量

    @staticmethod
    def conv_out_size_same(size, step):
        """Method for recording a benchmark directly.
        #   计算每一层反卷积的窗口大小，我是按照每次除以step的方法，无实际算法意义。
        #   Args:
        #       size: 图片的大小。
        #       step: 分段。
        """
        return int(math.ceil(float(size) / float(step)))

    @staticmethod
    def linear(x, output_size, weight_mean= 0.0, weight_stddev= 0.02, bias_init=0.0, name='Linear'):
        """Method for recording a benchmark directly.
        #   使用Tensorflow计算线性y = xw +b
        #   Args:
        #       :param x: 输入数据。shape is (height, weight)。
        #       :param output_size: 输出数据的维度
        #       :param weight_mean: 权重的平均值
        #       :param weight_stddev: 权重的方差
        #       :param bias_init: 偏移初始化
        #       :param name: 该线性变换的名称，为Tensorflow可视化做准备
        #   Return:
        #       :return y: y = xw +b    shape is (height, output_size)
        #       :return w:              shape is (weight, output_size)
        #       :return b:              shape is (output_size)
        """
        with tf.variable_scope(name):
            w = tf.get_variable("w", [x.shape[1], output_size], tf.float32, tf.random_normal_initializer(mean=weight_mean, stddev=weight_stddev))
            b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(bias_init))
            y = tf.matmul(x, w) + b
            return y, w, b

    @staticmethod
    def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='Batch_Normalizer'):
        """
        经过封装的BN层
        tensorflow中batch normalization的实现主要有下面三个：
            tf.nn.batch_normalization
            tf.layers.batch_normalization 建议
            tf.contrib.layers.batch_norm 建议
        封装程度逐个递进，建议使用tf.layers.batch_normalization或tf.contrib.layers.batch_norm
        :param x: 输入数据。shape is (n, height, weight, channel)。
        :param epsilon:
        :param momentum:
        :param train:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                                scale=True, is_training=train)

    @staticmethod
    def conv2d(images, output_dim, kernelsize=5, stddev=0.02, name="Conv2d"):
        with tf.variable_scope(name):
            # filter : [height, width, in_channels, output_channels]
            # 注意与转置卷积的不同
            filter_shape = [kernelsize, kernelsize, int(images.get_shape()[-1]), output_dim]
            # strides
            # 对应每一维的filter移动步长
            strides_shape = [1, 2, 2, 1]

            w = tf.get_variable('w', filter_shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(images, w, strides=strides_shape, padding='SAME')
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

            return conv

    @staticmethod
    def deconv2d(images, output_shape, kernelsize=5, stddev=0.02, name='Deconv2d'):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            # 注意与卷积的不同
            filter_shape = [kernelsize, kernelsize, output_shape[-1], images.get_shape()[-1]]
            # strides
            # 对应每一维的filter移动步长
            strides_shape = [1, 2, 2, 1]

            w = tf.get_variable('w', filter_shape, initializer=tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

            deconv = tf.nn.conv2d_transpose(images, w, output_shape=output_shape, strides=strides_shape)
            deconv = tf.nn.bias_add(deconv, b)

            return deconv, w, b

    def generator(self, noise_imgs, train=True):
        """Method for recording a benchmark directly.
        #   生成图片，说白了就是通过噪声反卷积。
        #   Args:
        #       noise_imgs: 单通道噪声图片.
        """
        with tf.variable_scope('generator'):
            # 分别对应每个layer的height, width
            s_h, s_w, _ = self.image_shape
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            # layer 0
            # 对输入噪音图片进行线性变换，由于是线性层级，所以还需要做形状重塑
            z, h0_w, h0_b = self.linear(noise_imgs, self.gf_size * 8 * s_h16 * s_w16)
            # reshape为合适的输入层格式，(-1, height, weight, 卷积通道个数)
            h0 = tf.reshape(z, [-1, s_h16, s_w16, self.gf_size * 8])
            # 对数据进行归一化处理 加快收敛速度
            h0 = self.batch_normalizer(h0, train=train, name='g_bn0')
            # 激活函数
            h0 = tf.nn.relu(h0)

            # layer 1
            # 卷积转置进行上采样 ==> BN 处理 ==> relu 下同。
            # input: h0, shape is (?, 3, 3, 512).
            # output: h1, shape is (64, 6, 6, 256).
            h1, h1_w, h1_b = self.deconv2d(h0, [self.each_batch_size, s_h8, s_w8, self.gf_size * 4], name='g_h1')
            h1 = self.batch_normalizer(h1, train=train, name='g_bn1')
            h1 = tf.nn.relu(h1)

            # layer 2
            h2, h2_w, h2_b = self.deconv2d(h1, [self.each_batch_size, s_h4, s_w4, self.gf_size * 2], name='g_h2')
            h2 = self.batch_normalizer(h2, train=train, name='g_bn2')
            h2 = tf.nn.relu(h2)

            # layer 3
            h3, h3_w, h3_b = self.deconv2d(h2, [self.each_batch_size, s_h2, s_w2, self.gf_size * 1], name='g_h3')
            h3 = self.batch_normalizer(h3, train=train, name='g_bn3')
            h3 = tf.nn.relu(h3)

            # layer 4
            h4, h4_w, h4_b = self.deconv2d(h3, [self.each_batch_size, s_h, s_w, 3], name='g_h4')
            return tf.nn.tanh(h4)

    def discriminator(self, real_imgs, reuse=False):
        """
        辨别器不复用。
        :param real_imgs: 真实图片
        :param reuse: 不复用
        :return:
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            # layer 0
            # 卷积操作
            # input: real_imgs, shape is (batch, height, weight, channels)
            # output: h0, shape is (batch, height, weight ,channels)
            h0 = self.conv2d(real_imgs, self.df_size, name='d_h0_conv')
            # 激活函数
            # 随着训练的进行，部分输入会落到硬饱和区，导致对应的权重无法更新。我们称之为“神经元死亡”
            # relu函数，x<0 部分就是硬饱和区，GAN在开始时失败率很高，会容易陷入其中，所以使用leaky relu
            # leak relu: f(x)=max(x, x*leak)
            h0 = tf.nn.leaky_relu(h0, alpha=0.2, name='d_h0_leak_relu')

            # layer 1
            h1 = self.conv2d(h0, self.df_size * 2, name='d_h1_conv')
            h1 = self.batch_normalizer(h1, name='d_bn1')
            h1 = tf.nn.leaky_relu(h1, alpha=0.2, name='d_h1_leak_relu')

            # layer 2
            h2 = self.conv2d(h1, self.df_size * 4, name='d_h2_conv')
            h2 = self.batch_normalizer(h2, name='d_bn2')
            h2 = tf.nn.leaky_relu(h2, alpha=0.2, name='d_h2_leak_relu')

            # layer 3
            h3 = self.conv2d(h2, self.df_size * 8, name='d_h3_conv')
            h3 = self.batch_normalizer(h3, name='d_bn3')
            h3 = tf.nn.leaky_relu(h3, alpha=0.2, name='d_h3_leak_relu')

            # layer 4
            h4, _, _ = self.linear(tf.reshape(h3, [self.each_batch_size, -1]), 1, name='d_h4_lin')

            return tf.nn.sigmoid(h4), h4

    @staticmethod
    def loss_graph(real_logits, fake_logits):
        # 生成器图片loss
        # 生成器希望判别器判断出来的标签为1
        gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
        # 判别器识别生成器图片loss
        # 判别器希望识别出来的标签为0
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        # 判别器识别真实图片loss
        # 判别器希望识别出来的标签为1
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
        # 判别器总loss
        dis_loss = tf.add(fake_loss, real_loss)
        return gen_loss, fake_loss, real_loss, dis_loss

    @staticmethod
    def optimizer_graph(gen_loss, dis_loss, learning_rate, beta1):
        # 所有定义变量
        train_vars = tf.trainable_variables()
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        # optimizer
        # 生成器与判别器作为两个网络需要分别优化
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(gen_loss,
                                                                                                  var_list=gen_vars)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(dis_loss,
                                                                                                  var_list=dis_vars)
        return gen_optimizer, dis_optimizer

    def train(self):
        # 真实图片
        real_imgs = tf.placeholder(tf.float32, self.each_batch_shape, name='real_images')
        # 噪声图片
        noise_imgs = tf.placeholder(tf.float32, [None, self.noise_img_size], name='noise_images')

        # 生成器图片
        fake_imgs = self.generator(noise_imgs)

        # 判别器
        real_outputs, real_logits = self.discriminator(real_imgs)
        fake_outputs, fake_logits = self.discriminator(fake_imgs, reuse=True)

        # 损失
        gen_loss, fake_loss, real_loss, dis_loss = self.loss_graph(real_logits, fake_logits)
        # 优化
        gen_optimizer, dis_optimizer = self.optimizer_graph(gen_loss, dis_loss, self.learning_rate, self.beta1)

        # 开始训练
        saver = tf.train.Saver()
        step = 0
        # 指定占用GPU比例
        # tensorflow默认占用全部GPU显存 防止在机器显存被其他程序占用过多时可能在启动时报错
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoch_size):
                batches = self.wives.get_batches_iter()
                for batch_imgs in batches:
                    # generator的输入噪声
                    noises = np.random.uniform(-1, 1, size=(self.each_batch_size, self.noise_img_size)).astype(np.float32)
                    # 优化
                    _ = sess.run(dis_optimizer, feed_dict={real_imgs: batch_imgs, noise_imgs: noises})
                    _ = sess.run(gen_optimizer, feed_dict={noise_imgs: noises})
                    _ = sess.run(gen_optimizer, feed_dict={noise_imgs: noises})
                    step += 1
                    print(datetime.datetime.now().strftime('%c'), epoch, step)
                    # 每一轮结束计算loss
                    # 判别器损失
                    loss_dis = sess.run(dis_loss, feed_dict={real_imgs: batch_imgs, noise_imgs: noises})
                    # 判别器对真实图片
                    loss_real = sess.run(real_loss, feed_dict={real_imgs: batch_imgs, noise_imgs: noises})
                    # 判别器对生成器图片
                    loss_fake = sess.run(fake_loss, feed_dict={real_imgs: batch_imgs, noise_imgs: noises})
                    # 生成器损失
                    loss_gen = sess.run(gen_loss, feed_dict={noise_imgs: noises})

                    print(datetime.datetime.now().strftime('%c'), ' epoch:', epoch, ' step:', step, ' loss_dis:', loss_dis,
                      ' loss_real:', loss_real, ' loss_fake:', loss_fake, ' loss_gen:', loss_gen)

            model_path = os.getcwd() + os.sep + "wifeface.model"
            saver.save(sess, model_path, global_step=step)

    def gen(self):
        # 生成图片
        noise_imgs = tf.placeholder(tf.float32, [None, self.noise_img_size], name='noise_imgs')
        sample_imgs = self.generator(noise_imgs, train=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            sample_noise = np.random.uniform(-1, 1, size=(self.sample_size, self.noise_img_size))
            samples = sess.run(sample_imgs, feed_dict={noise_imgs: sample_noise})
        for num in range(len(samples)):
            self.wives.save_image(samples[num], 'samples' + os.sep + str(num) + '.jpg')
