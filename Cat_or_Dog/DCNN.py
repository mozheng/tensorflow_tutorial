import tensorflow as tf


tf.flags.DEFINE_bool('debug', 'True', 'Debug mode: True/False' )

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_integer('batchsize','8','trainning batchsize')    #参数 默认值  说明
tf.flags.DEFINE_float('learning_rate','1e-4','learning_rate')
tf.flags.DEFINE_bool('train', "True", "Debug mode: True/ False")



def VGG_Net(images):
    layers = [
        'conv1_1_64', 'relu1_1', 'conv1_2_64', 'relu1_2', 'pool1_64',

        'conv2_1_128', 'relu2_1', 'conv2_2_128', 'relu2_2', 'pool2_64',

        'conv3_1_256', 'relu3_1', 'conv3_2_256', 'relu3_2', 'conv3_3_256',
        'relu3_3', 'conv3_4_256', 'relu3_4', 'pool3_64',

        'conv4_1_512', 'relu4_1', 'conv4_2_512', 'relu4_2', 'conv4_3_512',
        'relu4_3', 'conv4_4_512', 'relu4_4', 'pool4_64',

        'conv5_1_512', 'relu5_1', 'conv5_2_512', 'relu5_2', 'conv5_3_512',
        'relu5_3', 'conv5_4_512', 'relu5_4'
    ]

    current = images
    net = {}
    for i, name in enumerate(layers):
        layer_name = name[:4]
        if layer_name == 'conv':
            num = int(name.split('_')[-1])
            current = tf.nn.conv2d(input=current, filter=[num, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name=name)
        elif layer_name == 'relu':
            current = tf.nn.relu(current, name=name)
        elif layer_name == 'pool':
            num = int(name.split('_')[-1])
            current = tf.nn.max_pool(current, ksize=[num, 2, 2, 1], strides=[1, 2, 2, 1], name=name)
        net[name] = current
    return net


def loss_opt_function(image_net, labels):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=image_net, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.5).minimize(cost)
    return optimizer

def main(unused):
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    with tf.Graph().as_default() as graph:
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用90%显存
        with tf.Session(config=config) as sess:
            image_net = VGG_Net(images= )
            opt = loss_opt_function(image_net['relu5_4'],labels=)
            step = 0
            while step < 10000:
                step += 1
                _, loss, acc = sess.run([opt, cost, accuracy])
                if step % 1000 == 0:
                    print("step:",loss, acc)
            print("training finish!")





if __name__ == '__main__':
    tf.app.run()