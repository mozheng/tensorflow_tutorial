import tensorflow as tf


class SRGAN:
    def __init__(self):
        self.__batch_size__ = 64

    def deconv_layer(self, input, weights):
        # input_shape=[n,height,width,channel]
        input_shape = input.get_shape().as_list()
        # weights shape=[height,width,out_c,in_c]
        weights_shape = weights.get_shape().as_list()
        output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, weights_shape[2]]

        print("output_shape:", output_shape)

        deconv = tf.nn.conv2d_transpose(input, weights, output_shape=output_shape,
                                        strides=[1, 2, 2, 1], padding='SAME')
        return deconv

    def generator(self, x, is_training, reuse):
        with tf.variable_scope("generator", reuse=reuse):
            with tf.variable_scope("decovn1"):
                x = deconv_layer(x, [3, 3, 64, 3], [self.__batch_size__, 24, 24, 64], 1)
                x = tf.nn.relu(x)