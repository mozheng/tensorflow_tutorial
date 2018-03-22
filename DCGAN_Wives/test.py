# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 13:10
# @Author  : Mozheng
# @Email   : mozhengweiyi@163.com
# @Site    : 
# @File    : test.py
# @Software: PyCharm
# ---------------------------------------
# This file is focus on :
# 语言测试与学习
# ---------------------------------------

def a(x):
    for i in range(x):
        yield i

for i in a(11):
    print(i)