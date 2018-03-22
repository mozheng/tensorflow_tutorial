# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 12:54
# @Author  : Mozheng
# @Email   : mozhengweiyi@163.com
# @Site    : 
# @File    : main.py
# @Software: PyCharm python3.6
# ---------------------------------------
# This file is focus on :
#
# ---------------------------------------
from DCGANModel import DCGANModel

if __name__ == '__main__':
    avatar = DCGANModel()
    avatar.train()
