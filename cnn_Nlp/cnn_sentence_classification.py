#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 上午10:22
# @Author  : Aries
# @Site    : 
# @File    : cnn_sentence_classification.py
# @Software: PyCharm
'''
cnn实现句子分类--监督式学习
'''
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = 'http://cogcomp.org/Data/QA/QC/'
dir_name = 'question-classif-data'

data_list = []
count = []
words = []
dictionary = dict()

max_sent_length = 0
sent_length = 0

num_classes = 6
# all the type of question that are in the dataset
all_labels = ['NUM', 'LOC', 'HUM', 'DESC', 'ENTY', 'ABBR']


# 下载数据
def maybe_download(dir_name, filename, expected_bytes):
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
	if not os.path.exists(os.path.join(dir_name, filename)):
		filename, _ = urlretrieve(url + filename, os.path.join(dir_name, filename))
	print(os.path.join(dir_name, filename))
	statinfo = os.stat(os.path.join(dir_name, filename))
	if statinfo.st_size == expected_bytes:
		print('Found and verified %s' % os.path.join(dir_name, filename))
	else:
		print(statinfo.st_size)
		raise Exception('Failed to verify ' + os.path.join(dir_name, filename) + '. Can you get to it with a browser?')
	return filename


# 获取数据和标签
def read_data(filename):
	'''
	 Read data from a file with given filename
	 Returns a list of strings where each string is a lower case word
	'''
	# 最大的单词数
	global max_sent_length
	global sent_length
	questions = []
	labels = []
	with open(filename, 'r', encoding='latin-1') as f:
		for row in f:
			row_str = row.split(':')
			lb, q = row_str[0], row_str[1]
			q = q.lower()
			labels.append(lb)
			questions.append(q.split())
			if len(questions[-1]) > max_sent_length:
				max_sent_length = len(questions[-1])
				sent_length = max_sent_length
	return questions, labels


def build_dataset(questions):
	'''
	create a  dict:
	1.maps a string word to an ID & maps an ID to a string word
	2.List of list of (word,frequency)elements(eg.[(I,1),(to,2)...])
	3.Contain the string of text we read,where string words are replaced with word IDs
	:param questions:
	:return:
		data: list [[word,id]]
		count: list [[word,freq]]
		dictionary: dict [[word,id]]
		reverse_dictionary [[id,word]]
	'''
	for d in questions:
		words.extend(d)
	count.extend(collections.Counter(words).most_common())
	
	for word, _ in count:
		dictionary[word] = len(dictionary)
	
	for d in questions:
		data = []
		for word in d:
			index = dictionary[word]
			data.append(index)
		data_list.append(data)
	
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	
	return data_list, count, dictionary, reverse_dictionary


def cnn_layer(sent_inputs, w1, b1, w2, b2, w3, b3, w_fc1, b_fc1):
	h1_1 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w1, stride=1, padding='SAME') + b1)
	h1_2 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w2, stride=1, padding='SAME') + b2)
	h1_3 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w3, stride=1, padding='SAME') + b3)
	
	# Pooling over time operation
	
	# This is doing the max pooling. Thereare two options to do the max pooling
	# 1. Use tf.nn.max_pool operation on a tensor made by concatenating h1_1,h1_2,h1_3 and converting that tensor to 4D
	# (Because max_pool takes a tensor of rank >= 4 )
	# 2. Do the max pooling separately for each filter output and combine them using tf.concat
	# (this is the one used in the code)
	
	h2_1 = tf.reduce_max(h1_1, axis=1)
	h2_2 = tf.reduce_max(h1_2, axis=1)
	h2_3 = tf.reduce_max(h1_3, axis=1)
	
	h2 = tf.concat([h2_1, h2_2, h2_3], axis=1)
	
	# Calculate the fully connected layer output (no activation)
	# Note: since h2 is 2d [batch_size,number of parallel filters]
	# reshaping the output is not required as it usually do in CNNs
	logits = tf.matmul(h2, w_fc1) + b_fc1
	
	return logits


# 计算精度
def accuracy(lables, preds):
	return np.sum(np.argmax(lables, axis=1) == preds) / lables.shape[0]


class BatchGenerator(object):
	'''Generates a batch of data'''
	
	def __init__(self, batch_size, questions, labels, vocabulary_size):
		self.questions = questions
		self.labels = labels
		self.text_size = len(questions)
		self.batch_size = batch_size
		self.data_index = 0
		self.vocabulary_size = vocabulary_size
		assert len(self.questions) == len(self.labels)
	
	def generate_batch(self):
		global sent_length, num_classes
		global dictionary, all_labels
		
		inputs = np.zeros((self.batch_size, sent_length, self.vocabulary_size), dtype=np.float32)
		labels_ohe = np.zeros((self.batch_size, num_classes), dtype=np.float32)
		
		# check 是否可以抽样
		if self.data_index + self.batch_size > self.text_size:
			self.data_index = 0
		
		# 创建每个单词的独热编码表示
		for qi, que in enumerate(self.questions[self.data_index:self.data_index + self.batch_size]):
			for wi, word in enumerate(que):
				inputs[qi, wi, dictionary[word]] = 1.0
			# 将对应于特定类的索引设置为1 -- 独热表示
			labels_ohe[qi, all_labels.index(self.labels[self.data_index + qi])] = 1.0
		
		self.data_index = (self.data_index + self.batch_size) % self.text_size
		
		return inputs, labels_ohe
	
	def return_index(self):
		return self.data_index


def main():
	# 下载训练数据和测试数据
	filename = maybe_download(dir_name, 'train_1000.label', 60774)
	test_filename = maybe_download(dir_name, 'TREC_10.label', 23354)
	# 检查文件
	filenames = ['train_1000.label', 'TREC_10.label']
	num_files = len(filenames)
	for i in range(len(filenames)):
		file_exists = os.path.isfile(os.path.join(dir_name, filenames[i]))
		assert file_exists
	print('Files found and verified.')
	# 加载和准备数据 questions & labels
	for i in range(num_files):
		if i == 0:  # 训练数据
			train_questions, train_labels = read_data(os.path.join(dir_name, filenames[i]))
		if i == 1:  # 测试数据
			test_questions, test_labels = read_data(os.path.join(dir_name, filenames[i]))
	
	# Print some data to see everything is okey
	for j in range(5):
		print('train data')
		print('\ttrain Question %d: %s' % (j, train_questions[j]))
		print('\ttrain Label %d: %s\n' % (j, train_labels[j]))
		print('broadcast data')
		print('\tbroadcast Question %d: %s' % (j, test_questions[j]))
		print('\tbroadcast Label %d: %s\n' % (j, test_labels[j]))
		
		print('Max Sentence Length: %d' % max_sent_length)
	
	# 填充训练数据---扩展至句子最大单词数
	print('starting pading train_questions and broadcast questions')
	for qi, que in enumerate(train_questions):
		for _ in range(max_sent_length - len(que)):
			que.append('PAD')
		assert len(que) == max_sent_length
		train_questions[qi] = que
	# 填充测试数据
	for qi, que in enumerate(test_questions):
		for _ in range(max_sent_length - len(que)):
			que.append('PAD')
		assert len(que) == max_sent_length
		test_questions[qi] = que
	
	# 生成dictionary
	all_questions = list(train_questions)
	all_questions.extend(test_questions)
	
	all_question_ind, count, dictionary, reverse_dictionary = build_dataset(all_questions)
	vocabulary_size = len(dictionary)
	
	# 建模
	batch_size = 32
	# 在一个卷积层中使用不同的滤波器尺寸
	filter_sizes = [3, 5, 7]
	
	# input and labels
	sent_inputs = tf.placeholder(shape=[batch_size, sent_length, vocabulary_size], dtype=tf.float32, name='sent_inputs')
	sent_labels = tf.placeholder(shape=[batch_size, num_classes], dtype=tf.float32, name='sent_labels')
	
	# 初始化 网络层参数
	w1 = tf.Variable(tf.truncated_normal([filter_sizes[0], vocabulary_size, 1], stddev=0.2, dtype=tf.float32),
	                 name='w_1')
	b1 = tf.Variable(tf.truncated_normal([1], 0, 0.01, dtype=tf.float32), name='b_1')
	w2 = tf.Variable(tf.truncated_normal([filter_sizes[1], vocabulary_size, 1], stddev=0.2, dtype=tf.float32),
	                 name='w_2')
	b2 = tf.Variable(tf.truncated_normal([1], 0, 0.01, dtype=tf.float32), name='b_2')
	
	w3 = tf.Variable(tf.truncated_normal([filter_sizes[2], vocabulary_size, 1], stddev=0.2, dtype=tf.float32),
	                 name='w_3')
	b3 = tf.Variable(tf.truncated_normal([1], 0, 0.01, dtype=tf.float32), name='b_3')
	
	# 全连接层参数
	w_fc1 = tf.Variable(tf.truncated_normal([len(filter_sizes), num_classes], stddev=0.5, dtype=tf.float32),
	                    name='w_fc1')
	b_fc1 = tf.Variable(tf.truncated_normal([num_classes], 0, 0.01, dtype=tf.float32), name='b_fc1')
	logits = cnn_layer(sent_inputs, w1, b1, w2, b2, w3, b3, w_fc1, b_fc1)
	
	# 初始化loss函数
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=sent_labels, logits=logits))
	
	optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)
	
	# 预测
	predictions = tf.argmax(tf.nn.softmax(logits), axis=1)
	
	num_steps = 150
	with tf.Session() as session:
		tf.global_variables_initializer().run()
		train_gen = BatchGenerator(batch_size=batch_size, questions=train_questions, labels=train_labels,
		                           vocabulary_size=vocabulary_size)
		test_gen = BatchGenerator(batch_size=batch_size, questions=test_questions, labels=test_labels,
		                          vocabulary_size=vocabulary_size)
		# 多久计算一次精度
		test_interval = 1
		for step in range(num_steps):
			avg_loss = []
			for tr_i in range(len(train_questions) // batch_size - 1):
				train_input, train_label = train_gen.generate_batch()
				l, _ = session.run([loss, optimizer], feed_dict={sent_inputs: train_input, sent_labels: train_label})
				avg_loss.append(l)
			
			# 计算精度
			print('compute accuracy')
			test_accuracy = []
			if (step + 1) % test_interval == 0:
				for ts_i in range(len(train_questions) // batch_size - 1):
					test_input, test_label = test_gen.generate_batch()
					pred = session.run(predictions,
					                   feed_dict={sent_inputs: test_input, sent_labels: test_label})
					test_accuracy.append(accuracy(test_label, pred))
				
				print('Test accuracy at Epoch %d: %.3f' % (step, np.mean(test_accuracy) * 100.0))


if __name__ == '__main__':
	main()
