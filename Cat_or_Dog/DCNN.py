import tensorflow as tf
from .gen_tfrecord import Gen_TFrecord


tf.flags.DEFINE_bool('debug', 'True', 'Debug mode: True/False' )
tf.flags.DEFINE_integer('batchsize','8','trainning batchsize')    #参数 默认值  说明
tf.flags.DEFINE_float('learning_rate','1e-4','learning_rate')
tf.flags.DEFINE_bool('train', "True", "Debug mode: True/ False")
FLAGS=tf.flags.FLAGS


def head_Net(images):
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
    channels =3
    num = channels
    for i, name in enumerate(layers):
        layer_name = name[:4]
        if layer_name == 'conv':
            next_num = int(name.split('_')[-1])
            current = tf.nn.conv2d(input=current, filter=[3, 3, num, next_num], strides=[1, 1, 1, 1], padding='SAME', name=name)
            num = next_num
        elif layer_name == 'relu':
            current = tf.nn.relu(current, name=name)
        elif layer_name == 'pool':
            next_num = int(name.split('_')[-1])
            current = tf.nn.max_pool(current, ksize=[2, 2, num, next_num], strides=[1, 2, 2, 1], name=name)
            num = next_num
        net[name] = current
    return net


def loss_opt_function(image_net, labels):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=image_net, labels=labels))

    return cost


def VGG_Net(images,labels):
    net = head_Net(images)
    return loss_opt_function(net, labels)

def main(unused):
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)

    with tf.Graph().as_default() as graph:
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用90%显存
        gentfrecord = Gen_TFrecord()
        gentfrecord.make_list_file("../data/DogsvsCats/train", "train.txt")
        gentfrecord.generate_tfrecord_file("cat_dog.tfrecord")
        dataset = gentfrecord.get_batch_dataset("cat_dog.tfrecord")
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session(config=config) as sess:

            step = 0
            while step < 10000:
                try:
                    images, labels = sess.run(next_element)
                    cost = VGG_Net(images, labels)
                    optimizer = tf.train.AdamOptimizer(learning_rate=0.5).minimize(cost)
                    step += 1
                    _, loss, acc = sess.run([optimizer, cost])
                    if step % 1000 == 0:
                        print("step:", loss)

                except tf.errors.OutOfRangeError:
                    break

            print("training finish!")








if __name__ == '__main__':
    tf.app.run()