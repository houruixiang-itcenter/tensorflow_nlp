#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/19 下午5:43
# @Author  : Aries
# @Site    : 
# @File    : dnn_operation.py
# @Software: PyCharm
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python import learn
import car_dl.data_split as data_split
import car_dl.data_trans as data_trans
import numpy as np

import tensorflow as tf
import tensorboard

# todo 第一步 定义网络中的相关参数 --- 反向调节用
from tensorflow.python.framework import graph_util

is_train = False


def hidden_layer(input, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, layer_name):
	global is_train
	layer1 = tf.nn.leaky_relu(batch_norm_layer(value=tf.matmul(batch_norm_layer(input), w1) + b1))
	layer1 = layers.dropout(layer1, keep_prob=0.8, is_training=is_train)
	layer2 = tf.nn.leaky_relu(batch_norm_layer(tf.matmul(layer1, w2) + b2))
	layer2 = layers.dropout(layer2, keep_prob=0.8, is_training=is_train)
	layer3 = tf.nn.leaky_relu(batch_norm_layer(tf.matmul(layer2, w3) + b3))
	layer3 = layers.dropout(layer3, keep_prob=0.8, is_training=is_train)
	layer4 = tf.nn.leaky_relu(batch_norm_layer(tf.matmul(layer3, w4) + b4))
	layer4 = layers.dropout(layer4, keep_prob=0.8, is_training=is_train)
	output = tf.nn.leaky_relu(batch_norm_layer(value=tf.matmul(layer4, w5) + b5))
	return output


index = 0


def batch_sample(feature, label, batch_size):
	global index
	data_size = len(feature)
	num_batches_per_epoch = int(data_size / batch_size)
	batch_data = np.zeros((batch_size, 14), dtype=np.float32)
	batch_label = np.zeros((batch_size, 14), dtype=np.float32)
	shuffle_indices = np.random.permutation(np.arange(data_size))
	feature = feature[shuffle_indices]
	for i in range(batch_size):
		batch_data = feature[i:i + batch_size, :]
		batch_label = label[i:i + batch_size, :]
	if index >= num_batches_per_epoch * batch_size:
		index = 0
	else:
		index = (index + batch_size) % data_size
	
	return batch_data, batch_label


def batch_generator(all_data, batch_size, shuffle=True):
	"""
	:param all_data : all_data整个数据集
	:param batch_size: batch_size表示每个batch的大小
	:param shuffle: 每次是否打乱顺序
	:return:
	"""
	all_data = [np.array(d) for d in all_data]
	data_size = all_data[0].shape[0]
	print("data_size: ", data_size)
	if shuffle:
		p = np.random.permutation(data_size)
		all_data = [d[p] for d in all_data]
	
	batch_count = 0
	while True:
		if batch_count * batch_size + batch_size > data_size:
			batch_count = 0
			if shuffle:
				p = np.random.permutation(data_size)
				all_data = [d[p] for d in all_data]
		start = batch_count * batch_size
		end = start + batch_size
		batch_count += 1
		yield [d[start: end] for d in all_data]


# def batch_iter(sourceData_feature, sourceData_label, batch_size, num_epochs, shuffle=True):
# 	data_size = len(sourceData_feature)
#
# 	num_batches_per_epoch = int(data_size / batch_size)  # 样本数/batch块大小,多出来的“尾数”，不要了
#
# 	# for epoch in range(num_epochs):
# 	#     # Shuffle the data at each epoch
# 	#     if shuffle:
# 	#         shuffle_indices = np.random.permutation(np.arange(data_size))
# 	#         shuffled_data_feature = sourceData_feature[shuffle_indices]
# 	#         shuffled_data_label = sourceData_label[shuffle_indices]
# 	#     else:
# 	#         shuffled_data_feature = sourceData_feature
# 	#         shuffled_data_label = sourceData_label
# 	#
# 	#     for batch_num in range(num_batches_per_epoch):  # batch_num取值0到num_batches_per_epoch-1
# 	#         start_index = batch_num * batch_size
# 	#         end_index = min((batch_num + 1) * batch_size, data_size)
# 	#
# 	#         yield (shuffled_data_feature[start_index:end_index], shuffled_data_label[start_index:end_index])
#
# 	if shuffle:
# 		shuffle_indices = np.random.permutation(np.arange(data_size))
# 		shuffled_data_feature = sourceData_feature[shuffle_indices]
# 		shuffled_data_label = sourceData_label[shuffle_indices]
# 	else:
# 		shuffled_data_feature = sourceData_feature
# 		shuffled_data_label = sourceData_label
#
# 	for batch_num in range(num_batches_per_epoch):  # batch_num取值0到num_batches_per_epoch-1
# 		start_index = batch_num * batch_size
# 		end_index = min((batch_num + 1) * batch_size, data_size)


def batch_norm_layer(value, is_training=True, name='batch_norm'):
	'''
	批量归一化  返回批量归一化的结果

	args:
		value:代表输入，第一个维度为batch_size
		is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
			  默认测试模式
		name：名称。
	'''
	if is_training is True:
		# 训练模式 使用指数加权函数不断更新均值和方差
		return tf.contrib.layers.batch_norm(inputs=value, decay=0.99, updates_collections=None, is_training=True)
	else:
		# 测试模式 不更新均值和方差，直接使用
		return tf.contrib.layers.batch_norm(inputs=value, decay=0.99, updates_collections=None, is_training=False)


def main():
	batch_size = 100  # 设置小批量的size
	learning_rate = 0.1  # 设置初始学习率
	learning_rate_decay = 0.999  # 设置学习率的衰减
	max_steps = 1000000  # 最大训练步数
	features = 16
	
	# 获取数据
	train_data, train_label, vaild_data, vaild_label, test_data, test_label = data_split.get_all_datas_and_labels()
	
	'''
	定义训练轮数的变量 一般定义为不可训练的
	'''
	training_step = tf.Variable(0, dtype=tf.float32, trainable=True)
	
	w1 = tf.get_variable('w1', shape=[features, 100], initializer=layers.xavier_initializer(), dtype=tf.float32)
	# w2 = tf.Variable(tf.truncated_normal([8, 100], stddev=0.1))
	b1 = tf.Variable(tf.constant(0.1, shape=[100]))
	
	w2 = tf.Variable(tf.truncated_normal([100, 200], stddev=0.1))
	b2 = tf.Variable(tf.constant(0.1, shape=[200]))
	
	w3 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
	b3 = tf.Variable(tf.constant(0.1, shape=[100]))
	
	w4 = tf.Variable(tf.truncated_normal([100, 50], stddev=0.1))
	b4 = tf.Variable(tf.constant(0.1, shape=[50]))
	
	w5 = tf.get_variable('w4', shape=[50, 1], initializer=layers.xavier_initializer(), dtype=tf.float32)
	b5 = tf.Variable(tf.constant(0.1, shape=[1]))
	
	x = tf.placeholder(tf.float32, [None, features], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, 1], name='y-output')
	
	y = hidden_layer(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, 'y')
	y_pred = tf.nn.sigmoid(y)
	
	# # todo 滑动平均值
	# averages_class = tf.train.ExponentialMovingAverage(0.99, training_step)
	#
	# averages_op = averages_class.apply(tf.trainable_variables())
	#
	# average_y = hidden_layer(x, averages_class.average(w1), averages_class.average(b1),
	#                          averages_class.average(w2), averages_class.average(b2), averages_class.average(w3),
	#                          averages_class.average(b3), averages_class.average(w4), averages_class.average(b4),
	#                          'average_y')
	
	# todo 定义反向传播的参数
	print(y.shape)
	print(y_.shape)
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_)
	
	'''
	l2正则
	'''
	regularizer = layers.l2_regularizer(0.001)
	regularization = regularizer(w1) + regularizer(w2) + regularizer(w3) + regularizer(w4) + regularizer(
		b1) + regularizer(b2) + regularizer(b3) + regularizer(b4) + regularizer(b5)
	loss = tf.reduce_mean(cross_entropy) + regularization
	loss_summary = tf.summary.scalar('loss', loss)
	file_writer = tf.summary.FileWriter('logs/', tf.get_default_graph())
	
	learning_rate = tf.train.exponential_decay(learning_rate, training_step, len(train_data) / batch_size,
	                                           learning_rate_decay)
	
	train_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)
	
	with tf.control_dependencies([train_optimizer]):
		train_op = tf.no_op(name='predict')
	
	# todo 定义准确率
	# if y_pred > 0.5:
	# 	y_scord = 1
	# else:
	# 	y_scord = 0
	
	crorent_prediction = (y_pred >= 0.5)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(crorent_prediction, tf.float32), y_), tf.float32))
	
	# todo 执行阶段
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	vaild_train_feed = {x: train_data[:10000], y_: train_label[:10000]}
	validate_feed = {x: vaild_data, y_: vaild_label}
	test_feed = {x: test_data, y_: test_label}
	'''
	runing...-
	'''
	batch_gen = batch_generator([train_data, train_label], batch_size)
	
	for step in range(max_steps):
		global is_train
		is_train = True
		batch_data, batch_label = next(batch_gen)
		# batch_data, batch_label = next(iter)
		train_feed = {x: batch_data, y_: batch_label}
		_, loss1 = sess.run([train_op, loss], feed_dict=train_feed)
		if step % 100 == 0:
			summary_str = loss_summary.eval(feed_dict=validate_feed)
			file_writer.add_summary(summary_str, step)
		# if i % 500 == 0:
		#     validate_train_accuracy = sess.run(accuracy,
		#                                        feed_dict=vaild_train_feed)
		#     print('train!train!train!    After %d steps,validate_train_accuracy is %g%%' % (i, validate_train_accuracy * 100))
		if step % 1000 == 0:
			y_pre, validate_accuracy, validate_pre = sess.run([y_pred, accuracy, crorent_prediction],
			                                                  feed_dict=validate_feed)
			print('loss %f' % loss1)
			print('\n')
			print('After %d steps,vaild_train_accuracy is %g%%' % (step, validate_accuracy * 100))
	
	# 开始测试
	is_train = False
	test_accuracy = sess.run(accuracy, feed_dict=test_feed)
	print('After %d steps,test_accuracy is %g%%' % (max_steps, test_accuracy * 100))
	
	graph_def = tf.get_default_graph().as_graph_def()
	
	output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
	
	with tf.gfile.GFile('./' + 'model.pb', 'wb') as f:
		f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
	main()
