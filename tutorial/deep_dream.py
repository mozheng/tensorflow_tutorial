#coding: UTF-8

import numpy as np
import tensorflow as tf

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

model_file = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_file,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read()) # 二进制调用实际上是 ParseFromString()

    # 定义t_input为输入图片
t_input = tf.placeholder(tf.float32, name='input')
imagent_mean = 117.0

t_preprocessed = tf.expand_dims(t_input - imagent_mean ,0)
tf.import_graph_def(graph_def,{'input':t_preprocessed}) #将图从graph_def导入到当前默认图中, 它们将在未来的版本中被删除。

layers =[op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import' in op.name]
name = "mixed4d_3x3_bottleneck_pre_relu"
print(graph.get_tensor_by_name('import/'+name+':0').get_shape()) # 形如'conv1'是节点名称，而'conv1:0'是张量名称，表示节点的第一个输出张量

channel = 139 # select a channel
layout_output = graph.get_tensor_by_name('import/'+name+':0')

# 定义原始噪声
img_noise = np.random.uniform(low=1.0, high=254.0, size=(224, 224, 3))

def render_navie(t_obj, t_input, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

