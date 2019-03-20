import tensorflow as tf


tf.flags.DEFINE_bool('debug', 'True', 'Debug mode: True/False' )

FLAGS = tf.flags.FLAGS



def VGG_Net(batch, images):
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

def main(unused):
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    with tf.Graph().as_default() as graph:
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用90%显存
        with tf.Session(config=config) as sess:
            image_net = VGG_Net()
            conv_final_layer = image_net["conv5_3"]
            res = conv_final_layer





if __name__ == '__main__':
    tf.app.run()