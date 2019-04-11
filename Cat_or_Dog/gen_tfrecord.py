import os
import tensorflow as tf
import cv2
import random
import tensorflow.contrib.slim as slim


class Gen_TFrecord:
    img_ext = {".png", ".jpg", ".jpeg",
               ".PNG", "JPG", ".JPEG"}
    tfrecord_file = ""
    dataset_dir = ""
    trainfile = "train.txt"
    crop_size = (224, 224)
    total_size = 0
    def make_list_file(self, dataset_dir, trainfile="train.txt"):
        self.trainfile = trainfile
        self.dataset_dir = dataset_dir
        imageslist = []
        for filename in os.listdir(self.dataset_dir):
            if (os.path.splitext(filename)[-1] in self.img_ext):
                label = 0
                if "dog" == filename.split(".")[0]:
                    label = 1
                imageslist.append(os.path.join(self.dataset_dir, filename) + "," + str(label) + "\n")
        with open(self.trainfile, "w") as f:
            f.writelines(imageslist)
            self.total_size = len(imageslist)
        print("Listfile: " + self.trainfile + " is finished!")

    def __load_and_shuffle_images_listfile(self, imagesfile):
        imageslist = open(self.trainfile).readlines()
        imagesandlabel = []
        for line in imageslist:
            if "" == line or None == line:
                break
            filepath = line.split(",")[0]
            label = int(line.rstrip('\n').split(",")[1])
            imagesandlabel.append((filepath, label))
        random.shuffle(imagesandlabel)
        return imagesandlabel

    ##下面开始生成TFRecord
    def generate_tfrecord_file(self, tfrecord_file):
        self.tfrecord_file = tfrecord_file
        imagesandlabel = self.__load_and_shuffle_images_listfile(self.tfrecord_file)
        with tf.python_io.TFRecordWriter(self.tfrecord_file) as tfrecord_writer:
            for filepath, label in imagesandlabel:
                if "" == filepath or None == filepath:
                    break
                image = cv2.imread(filepath)

                if not image is None:
                    image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_CUBIC)
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                        'image_format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
                        'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
                        'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
                        'image_channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[2]])),
                        'image_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    }))
                    tfrecord_writer.write(example.SerializeToString())  # 序列化为字符串

        print("TFrecord file:" + self.tfrecord_file + "  is finished!")

    #### 下面是解析
    def __parse(self, example):
        tf_record_features = tf.parse_single_example(
            example, features={
                'image_raw': tf.FixedLenFeature([], tf.string, default_value=''),
                'image_format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
                'image_height': tf.FixedLenFeature([], tf.int64, default_value=0),
                'image_width': tf.FixedLenFeature([], tf.int64, default_value=0),
                'image_channel': tf.FixedLenFeature([], tf.int64, default_value=0),
                'image_label': tf.FixedLenFeature([], tf.int64, default_value=0)
            }
        )
        h = tf.cast(tf_record_features['image_height'], tf.int32)
        w = tf.cast(tf_record_features['image_width'], tf.int32)
        c = tf.cast(tf_record_features['image_channel'], tf.int32)
        image = tf.reshape(tf.decode_raw(tf_record_features['image_raw'], tf.uint8), [h, w, c])
        image = tf.cast(image, tf.float32)
        label = tf.cast(tf_record_features['image_label'], tf.int32)
        return image, label


    def get_batch_dataset(self, record_file, shuffle_size=50, repeat_size=1, batch_size=32):
        num_threads = tf.constant(1, dtype=tf.int32)
        # num_parallel_calls用多个线程解析
        # 每次shuffle的大小
        # 当repeat()中没有参数,表示无限的重复数据集（训练集中有55000个样本）.但repeat(2)时,相当于重复2遍数据集（即epoch=2)
        # 这里选择无限重复数据集
        # batch(55)表示batch_size = 55
        dataset = tf.data.TFRecordDataset(record_file).map(
            self.__parse, num_parallel_calls=num_threads).shuffle(shuffle_size).repeat(repeat_size).batch(batch_size)

        return dataset

