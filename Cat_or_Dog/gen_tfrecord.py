import os
import tensorflow as tf
import numpy as np
import cv2
import random


dataset_dir = '/root/cloud_annation/pcs/100-1-0301'
record_file = 'train.tfrecod'



class Gen_TFrecord_Factory:
    img_ext = {".png", ".jpg", ".jpeg",
               ".PNG", "JPG", ".JPEG"}
    def make_list_file(self, path):
        imageslist = []
        for filename in os.listdir(path):
            if (os.path.splitext(filename)[-1] in img_ext):
                label = 0
                if "dog" == filename.split(".")[0]:
                    label = 1
                imageslist.append(filename + "," + str(label)+"\n")
        with open("train.txt","w") as f:
            f.writelines(imageslist)

    def __load_and_shuffle_images_listfile(self, imagesfile):
        imageslist = open(imagesfile).readlines()
        imagesandlabel =[]
        for line in imageslist:
            if ""== line or None == line:
                break
            filepath = line.split(",")[0]
            label = int(line.rstrip('\n').split(",")[1])
            imagesandlabel.append((filepath, label))
        random.shuffle(imagesandlabel)
        return imagesandlabel


##下面开始生成TFRecord
    def generate_tfrecord_file(self, imagesfile):
        imagesandlabel = self.__load_and_shuffle_images_listfile(imagesfile)
        with tf.python_io.TFRecordWriter(record_file) as tfrecord_writer:
            for filename, label in imagesandlabel:
                if ""== filename or None == filename:
                    break
                image = cv2.imread(os.path.join(dataset_dir, filename))
                example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'image_heigh': tf.train.Feature(int64_list =tf.train.Int64List(value=[image.shape[0]])),
                'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
                'image_channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[2]])),
                'image_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
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