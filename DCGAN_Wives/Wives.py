# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 10:06
# @Author  : Mozheng
# @Email   : mozhengweiyi@163.com
# @Site    : 
# @File    : Wives.py
# @Software: PyCharm python3.6
# ---------------------------------------
# This file is focus on :
#
# ---------------------------------------

import os
import numpy as np
import cv2
import glob


class Wives:
    # wives，wife的复数。都是纯洁的小朋友，不熟悉也正常。毕竟单数都没有。
    def __init__(self, path=os.getcwd()):
        """Method for recording a benchmark directly.
        # 初始化数据集三大哲学问题:
        # 你是谁？ 名字，大小
        # 从哪来？ 路径，列表
        # 到哪去？ 要做什么
        """
        self.dataset_name = "faces"     # 数据集目录中，数据文件夹的名称
        self.sum = 0
        self.image_shape = (96, 96, 3)
        self.image_extension = "jpg"

        self.dataset_path = path
        self.__image_list = self._get_images_list(self.dataset_path)
        self.all_images_sum = self.get_images_sum()

        # 这里一定要用each，all命名，否则自己就懵了。
        self.each_batch_size = 64
        self.each_batch_shape = (self.each_batch_size,) + self.image_shape
        self.all_batch_sum = self.all_images_sum // self.each_batch_size

        # 下面为数据预处理，图片重新裁了一下
        self.resize_shape = (48, 48, 3)
        self.crop = False
        self.image_shape = self.image_shape if not self.crop else self.resize_shape


    def _get_images_list(self, path):
        """
        返回图片列表
        """
        return glob.glob(os.path.join(path, self.dataset_name, "*."+self.image_extension))

    def get_images_sum(self):
        """
        返回图片的总数量
        """
        return len(self.__image_list)

    def _get_image(self, filepath):
        """
        :param filepath: 图片的文件路径
        :return: 处理好的opencv图片格式
        返回处理好的图像
        """
        img = cv2.imread(filepath).astype(np.float32)
        return cv2.resize(img, self.image_shape) if self.crop else img

    def get_batches_iter(self):
        """
        每次返回一个batch的迭代器。
        """
        start = 0
        end = self.each_batch_size
        for _ in range(self.all_batch_sum):
            name_list = self.__image_list[start:end]
            imgs = [self._get_image(name) for name in name_list]
            batches = np.zeros(self.each_batch_shape)
            batches[::] = imgs
            yield batches
            start += self.each_batch_size
            end += self.each_batch_size


