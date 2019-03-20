import os
import tensorflow as tf
import numpy as np
import cv2
import random


dataset_dir = '/root/cloud_annation/pcs/100-1-0301'
record_file = 'train.tfrecod'

img_ext = {".png",
           ".jpg",
           ".jpeg",
           ".PNG",
           "JPG",
           ".JPEG",
        }

def load_and_shuffle_images_listfile(imagesfile):
    imageslist = open(imagesfile).writelines()
    return random.shuffle(imageslist)
##下面开始生成TFRecord


with tf.python_io.TFRecordWriter(record_file) as tfrecord_writer:
    for filename in os.listdir(dataset_dir):
        if (os.path.splitext(filename)[-1] in img_ext):
            image = cv2.imread(os.path.join(dataset_dir, filename))
            example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np_points_clould.tobytes()])),
            'image_heigh': tf.train.Feature(int64_list =tf.train.Int64List(value=[np.shape(np_points_clould)[0]])),
            'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.shape(np_points_clould)[0]])),
            'image_channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.shape(np_points_clould)[0]])),
            'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.shape(np_points_clould)[0]])),
            }))
            tfrecord_writer.write(example.SerializeToString())  # 序列化为字符串


#### 下面是解析
def get_record_parser():
    def parse(example):
        features = tf.parse_single_example(
            example, features={
                'pcs_raw': tf.FixedLenFeature([], tf.string, default_value=''),
                'len': tf.FixedLenFeature([], tf.int64, default_value=0)
            }
        )
        pcs = tf.reshape(tf.decode_raw(features['pcs_raw'], tf.float32), [-1,4])
        return pcs

    return parse


def get_batch_dataset(record_file, parser):
    num_threads = tf.constant(5, dtype=tf.int32)
    # num_parallel_calls用多个线程解析
    # 每次shuffle的大小
    # 当repeat()中没有参数,表示无限的重复数据集（训练集中有55000个样本）.但repeat(2)时,相当于重复2遍数据集（即epoch=2)
    # 这里选择无限重复数据集
    # batch(55)表示batch_size = 55
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(1000).repeat().batch(55)

    return dataset


parser = get_record_parser()
dataset = get_batch_dataset(record_file, parser)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(next_element)
    print(next_element)