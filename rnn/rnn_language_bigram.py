#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 上午11:22
# @Author  : Aries
# @Site    : 
# @File    : rnn_language_bigram.py
# @Software: PyCharm
'''文本生成通话故事'''
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
from scipy.sparse import lil_matrix
import ssl

# import nltk
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://www.cs.cmu.edu/~spok/grimmtmp/'
dir_name = 'stories'
num_files = 100
documents = []

chars = []
data_list = []
count = []
dictionary = dict({'UNK': 0})
reverse_dictionary = dict()
vocabulary_size = 0


# 下载数据
def maybe_download(filename):
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
	if not os.path.exists(dir_name + os.sep + filename):
		filename, _ = urlretrieve(url + filename, dir_name + os.sep + filename)
	else:
		print('File ', filename, ' already exists.')
	return filename


# 读取数据
def read_data(filename):
	with open(filename) as f:
		data = tf.compat.as_str(f.read())
		data = data.lower()
		data = list(data)
	return data


def build_dataset(documents):
	'''
	create a  dict:
	1.maps a string word to an ID & maps an ID to a string word
	2.List of list of (word,frequency)elements(eg.[(I,1),(to,2)...])
	3.Contain the string of text we read,where string words are replaced with word IDs
	:param questions:
	:return:
		chars: list [[word,id]]
		count: list [[word,freq]]
		dictionary: dict [[word,id]]
		reverse_dictionary [[id,word]]
	'''
	for d in documents:
		chars.extend(d)
	count.extend(collections.Counter(chars).most_common())
	for char, c in count:
		# Only add a bigram to dictionary if its frequency is more than 10
		if c > 10:
			dictionary[char] = len(dictionary)
	unk_count = 0
	for d in documents:
		data = []
		for char in d:
			if char in dictionary:
				index = dictionary[char]
			else:
				index = dictionary['UNK']
				unk_count += 1
			data.append(index)
		data_list.append(data)
	
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	
	return data_list, count, dictionary, reverse_dictionary


class DataGeneratorOHE(object):
	
	def __init__(self, text, batch_size, num_unroll):
		self._text = text
		self._text_size = len(self._text)
		self._batch_size = batch_size
		self._num_unroll = num_unroll
		self._segment = self._text_size // self._batch_size
		# 我们把文本分成几段，然后从一段中抽取一个条目来对这批数据进行采样
		self._cursor = [offset * self._segment for offset in range(self._batch_size)]
	
	def next_batch(self):
		
		batch_data = np.zeros((self._batch_size, vocabulary_size), dtype=np.float32)
		batch_labels = np.zeros((self._batch_size, vocabulary_size), dtype=np.float32)
		
		for b in range(self._batch_size):
			if self._cursor[b] + 1 >= self._text_size:
				self._cursor[b] = b * self._segment
			
			batch_data[b, self._text[self._cursor[b]]] = 1.0
			batch_labels[b, self._text[self._cursor[b] + 1]] = 1.0
			self._cursor[b] = (self._cursor[b] + 1) % self._text_size
		
		return batch_data, batch_labels
	
	def unroll_batches(self):
		unroll_data, unroll_labels = [], []
		for ui in range(self._num_unroll):
			data, labels = self.next_batch()
			unroll_data.append(data)
			unroll_labels.append(labels)
		return unroll_data, unroll_labels
	
	def reset_indices(self):
		self._cursor = [offset * self._segment for offset in range(self._batch_size)]


def sample(distribution):
	# 从预测分布中抽取一个单词
	best_idx = np.argmax(distribution)
	return best_idx


def main():
	filenames = [format(i, '03d') + '.txt' for i in range(1, 101)]
	for i in filenames:
		maybe_download(i)
	global documents
	for i in range(num_files):
		print('\nProcessing file %s' % os.path.join(dir_name, filenames[i]))
		chars = read_data(os.path.join(dir_name, filenames[i]))
		two_grams = [''.join(chars[ch_i:ch_i + 2]) for ch_i in range(0, len(chars) - 2, 2)]
		documents.append(two_grams)
		print('Data size (Characters) (Document %d) %d' % (i, len(two_grams)))
		print('Sample string (Document %d) %s' % (i, two_grams[:50]))
	global data_list, count, dictionary, reverse_dictionary, vocabulary_size
	data_list, count, dictionary, reverse_dictionary = build_dataset(documents)
	vocabulary_size = len(dictionary)
	
	# 初始化参数
	tf.reset_default_graph()
	# 定义展开的数量
	num_unroll = 50
	batch_size = 64
	test_batch_size = 1
	
	hidden = 64
	
	in_size, out_size = vocabulary_size, vocabulary_size
	
	# todo 初始化data和labels  与num_unroll相关  [num_unroll,batch_size,in_size] && [num_unroll,batch_size,out_size]
	train_dataset, train_labels = [], []
	for ui in range(num_unroll):
		train_dataset.append(tf.placeholder(tf.float32, shape=[batch_size, in_size], name='train_dataset_%d' % ui))
		train_labels.append(tf.placeholder(tf.float32, shape=[batch_size, out_size], name='train_lables_%d' % ui))
	
	valid_dataset = tf.placeholder(tf.float32, shape=[1, in_size], name='valid_dataset')
	valid_labels = tf.placeholder(tf.float32, shape=[1, in_size], name='valid_labels')
	
	test_dataset = tf.placeholder(tf.float32, shape=[test_batch_size, in_size], name='test_dataset')
	
	# todo 定义网络层参数
	# x-h  对应U
	w_xh = tf.Variable(tf.truncated_normal([in_size, hidden], stddev=0.2, dtype=tf.float32), name='w_xh')
	# h-h  对应W
	w_hh = tf.Variable(tf.truncated_normal([hidden, hidden], stddev=0.2, dtype=tf.float32), name='w_hh')
	# h-y
	w_hy = tf.Variable(tf.truncated_normal([hidden, out_size], stddev=0.2, dtype=tf.float32), name='w_hy')
	
	# todo 上一时间序列的隐藏层输出,也就是 RNN的记忆
	prev_train_h = tf.Variable(tf.zeros([batch_size, hidden], dtype=tf.float32), name='train_h', trainable=False)
	
	prev_valid_h = tf.Variable(tf.zeros([1, hidden], dtype=tf.float32), name='valid_h', trainable=False)
	
	prev_test_h = tf.Variable(tf.zeros([test_batch_size, hidden], dtype=tf.float32), name='text_h', trainable=False)
	
	# 定义RNN的网络
	y_scores, y_pre = [], []
	# 记录rnn中每个时刻的输出
	outputs = list()
	
	output_h = prev_train_h
	
	# todo 计算当前的输出   abcd ->bcde  计算每一个h
	for ui in range(num_unroll):
		output_h = tf.nn.tanh(tf.matmul(tf.concat([train_dataset[ui], output_h], 1), tf.concat([w_xh, w_hh], 0)))
		outputs.append(output_h)
	
	# todo 输出 单个rnn网络的最终输出
	y_scores = [tf.matmul(outputs[ui], w_hy) for ui in range(num_unroll)]
	y_pre = [tf.nn.softmax(y_scores[ui]) for ui in range(num_unroll)]
	
	train_perplexity_without_exp = tf.reduce_sum(tf.concat(train_labels, 0) * -tf.log(tf.concat(y_pre, 0) + 1e-10)) / (
			num_unroll * batch_size)
	
	next_valid_state = tf.nn.tanh(tf.matmul(valid_dataset, w_xh) +
	                              tf.matmul(prev_valid_h, w_hh))
	with tf.control_dependencies([tf.assign(prev_valid_h, next_valid_state)]):
		valid_scores = tf.matmul(next_valid_state, w_hy)
		valid_pre = tf.nn.softmax(valid_scores)
	
	# Validation data related perplexity
	# todo 计算困惑度
	valid_perplexity_without_exp = tf.reduce_sum(valid_labels * -tf.log(valid_pre + 1e-10))
	
	# Calculating hidden output for broadcast data
	next_test_state = tf.nn.tanh(tf.matmul(test_dataset, w_xh) +
	                             tf.matmul(prev_test_h, w_hh)
	                             )
	
	# Making sure that the broadcast hidden state is updated
	# every time we make a prediction
	# todo 执行完tf.assign(prev_test_h, next_test_state)之后才可以执行下面操作
	with tf.control_dependencies([tf.assign(prev_test_h, next_test_state)]):
		test_pre = tf.nn.softmax(tf.matmul(next_test_state, w_hy))
	
	# todo 计算loss,交叉熵   tf.assign(prev_train_h, output_h) 为下一个batch初始的h赋值
	with tf.control_dependencies([tf.assign(prev_train_h, output_h)]):
		rnn_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.concat(y_scores, 0),
		                                                                     labels=tf.concat(train_labels, 0)))
	rnn_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	
	# 将得到的梯度运用到所有的可训练参数,进行梯度下降
	gradients, v = zip(*rnn_optimizer.compute_gradients(rnn_loss))
	# todo 使用梯度裁剪防止梯度爆炸
	gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
	rnn_optimizer = rnn_optimizer.apply_gradients(zip(gradients, v))
	
	# todo 重置隐藏层状态,防止测试时候重用
	reset_train_h_op = tf.assign(prev_train_h, tf.zeros([batch_size, hidden], dtype=tf.float32))
	reset_valid_h_op = tf.assign(prev_valid_h, tf.zeros([1, hidden], dtype=tf.float32))
	reset_test_h_op = tf.assign(prev_test_h, tf.zeros([test_batch_size, hidden], dtype=tf.float32))
	
	# start training
	num_steps = 26
	# todo 在一个步骤中为每个文档执行多少个训练步骤
	steps_per_document = 100
	# 多久执行一次验证步骤
	valid_summary = 1
	
	# todo 我们运行的测试集是20和100
	train_doc_count = 20
	# Number of docs we use in a single step
	# When train_doc_count = 20 => train_docs_to_use = 5
	# # When train_doc_count = 100 => train_docs_to_use = 10
	train_docs_to_use = 5
	
	# Store the training and validation perplexity at each step
	valid_perplexity_ot = []
	train_perplexity_ot = []
	
	session = tf.InteractiveSession()
	
	tf.global_variables_initializer().run()
	
	print('Init...')
	average_loss = 0
	
	# 确定前10份文件
	long_doc_ids = []
	for di in range(num_files):
		if len(data_list[di]) > (num_steps + 1) * steps_per_document:
			long_doc_ids.append(di)
		if len(long_doc_ids) == 10:
			break
	
	# 生成验证数据
	data_gens = []
	valid_gens = []
	for fi in range(num_files):
		if fi not in long_doc_ids:
			data_gens.append(DataGeneratorOHE(data_list[fi], batch_size, num_unroll))
		else:
			data_gens.append(DataGeneratorOHE(data_list[fi][:-steps_per_document], batch_size, num_unroll))
			valid_gens.append(DataGeneratorOHE(data_list[fi][-steps_per_document:], 1, 1))
	feed_dict = {}
	for step in range(num_steps):
		print('\n')
		# todo 随机抽取文档
		for di in np.random.permutation(train_doc_count)[:train_docs_to_use]:
			doc_perplexity = 0
			for doc_step_id in range(steps_per_document):
				u_data, u_labels = data_gens[di].unroll_batches()
				for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
					feed_dict[train_dataset[ui]] = dat
					feed_dict[train_labels[ui]] = lbl
				
				_, l, step_predictons, _, step_labels, step_perplexity = session.run(
					[rnn_optimizer, rnn_loss, y_pre, train_dataset, train_labels, train_perplexity_without_exp],
					feed_dict=feed_dict)
				doc_perplexity += step_perplexity
				average_loss += step_perplexity
			
			print('Document %d step %d processed (Perplecity: %.2f).' % (
				di, step + 1, np.exp(doc_perplexity / steps_per_document)))
			session.run(reset_train_h_op)
		
		# vaildation phase
		if step % valid_summary == 0:
			# compute average loss
			average_loss = average_loss / (train_docs_to_use * steps_per_document * valid_summary)
			print('Average loss  at step %d: %f' % (step + 1, average_loss))
			print('\tPreplwxity at step %d: %f ' % (step + 1, np.exp(average_loss)))
			train_perplexity_ot.append(np.exp(average_loss))
			
			# reset loss
			average_loss = 0
			valid_loss = 0
			for v_doc_id in range(10):
				for v_step in range(steps_per_document // 2):
					uv_data, uv_lanbels = valid_gens[v_doc_id].unroll_batches()
					v_prep = session.run(valid_perplexity_without_exp,
					                     feed_dict={valid_dataset: uv_data[0], valid_labels: uv_lanbels[0]})
					valid_loss += v_prep
				session.run(reset_valid_h_op)
				valid_gens[v_doc_id].reset_indices()
			print()
			v_perplexity = np.exp(valid_loss / (steps_per_document * 10.0 // 2))
			print('Vaild Perplexity: %.2f\n' % v_perplexity)
			valid_perplexity_ot.append(v_perplexity)
			
			print('Generated Text after epoch %d ...' % step)
			segments_to_generate = 1
			chars_in_segment = 1000
			
			for _ in range(segments_to_generate):
				print('======================== New Text Segment =======================')
				test_word = np.zeros((1, in_size), dtype=np.float32)
				test_word[0, data_list[np.random.randint(0, num_files)][np.random.randint(0, 100)]] = 1.0
				print('\t', reverse_dictionary[np.argmax(test_word[0])], end='')
				
				for _ in range(chars_in_segment):
					test_data, test_prev,test_state = session.run([test_dataset, prev_test_h,next_test_state], feed_dict={test_dataset: test_word})
					next_data_ind = sample(test_data.ravel())
					next_prev_ind = sample(test_prev.ravel())
					next_state_ind = sample(test_state.ravel())
					a = reverse_dictionary[next_data_ind]
					b = reverse_dictionary[next_prev_ind]
					c = reverse_dictionary[next_state_ind]
					test_pred = session.run(test_pre, feed_dict={test_dataset: test_word})
					next_ind = sample(test_pred.ravel())
					test_word = np.zeros((1, in_size), dtype=np.float32)
					test_word[0, next_ind] = 1.0
					print(reverse_dictionary[next_ind], end='')
				print('')
				session.run(reset_test_h_op)
				print('================================================================')
			print('')
	
	np.save('./stories/train_perplexity_ot', train_perplexity_ot)
	np.save('./stories/valid_perplexity_ot', valid_perplexity_ot)
			
	x_axis = np.arange(len(train_perplexity_ot[1:25]))
	f, (ax1, ax2) = pylab.subplots(1, 2, figsize=(18, 6))
			
	ax1.plot(x_axis, train_perplexity_ot[1:25], label='Train')
	ax2.plot(x_axis, valid_perplexity_ot[1:25], label='Valid')
			
	pylab.title('Train and Valid Perplexity over Time', fontsize=24)
	ax1.set_title('Train Perplexity', fontsize=20)
	ax2.set_title('Valid Perplexity', fontsize=20)
	ax1.set_xlabel('Epoch', fontsize=20)
	ax2.set_xlabel('Epoch', fontsize=20)
	pylab.savefig('RNN_perplexity.png')
	pylab.show()
	
	
	
	
	
	
			


if __name__ == '__main__':
	main()
