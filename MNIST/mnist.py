# -*- coding: utf-8 -*-

import time
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

''' 
权重w和偏置b
初始化为一个接近0的很小的正数
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1) # 截断正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) # 常量0.1
    return tf.Variable(initial)

'''
卷积和池化，卷积步长为1（stride size），0边距（padding size）
池化用简单传统的2x2大小的模板max pooling
'''
def conv2d(x, W):
    # strides[1,,,1]默认为1，中间两位为size，padding same为0，保证输入输出大小一致
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')

# 计算开始时间
start = time.clock()
# MNIST数据输入
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 图像输入输出向量
x = tf.placeholder(tf.float32, [None, 784]) 
y_ = tf.placeholder(tf.float32, [None,10])

# 第一层，由一个卷积层加一个maxpooling层
# 卷积核的大小为5x5，个数为32
# 卷积核张量形状是[5, 5, 1, 32]，对应size，输入通道为1，输出通道为32
# 每一个输出通道都有一个对应的偏置量
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
x_image = tf.reshape(x, [-1, 28, 28, 1]) # -1代表None
# x_image权重向量卷积，加上偏置项，之后应用ReLU函数，之后进行max_polling
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层，结构不变，输入32个通道，输出64个通道
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# 全连接层
'''
图片尺寸变为7x7（28/2/2=7），加入有1024个神经元的全连接层，把池化层输出张量reshape成向量
乘上权重矩阵，加上偏置，然后进行ReLU
'''
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout，用来防止过拟合
# 加在输出层之前，训练过程中开启dropout，测试过程中关闭
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层, 添加softmax层，类别数为10
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 训练和评估模型
'''
ADAM优化器来做梯度最速下降，feed_dict加入参数keep_prob控制dropout比例
'''
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  # 计算交叉熵
# 使用adam优化器来以0.0001的学习率来进行微调
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 判断预测标签和实际标签是否匹配
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 启动创建的模型，并初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 开始训练模型，循环训练300次
for i in range(300):
	batch = mnist.train.next_batch(50) # batch 大小设置为50
	if i%10 == 0:
		train_accuracy = accuracy.eval(session=sess,
                                 feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
		print("step %d, train_accuracy %g" %(i,train_accuracy))
	# 神经元输出保持keep_prob为0.5，进行训练
	train_step.run(session=sess, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

# 神经元输出保持keep_prob为1.0，进行测试
print("test accuracy %g" %accuracy.eval(session=sess,
                                        feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
	
# 计算程序结束时间
end = time.clock()
print("running time is %g s" %(end-start))