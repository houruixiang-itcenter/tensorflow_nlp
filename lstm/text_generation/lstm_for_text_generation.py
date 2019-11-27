#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/22 下午2:38
# @Author  : Aries
# @Site    :
# @File    : lstm_for_text_generation.py
# @Software: PyCharm


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
import csv
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://www.cs.cmu.edu/~spok/grimmtmp/'
num_files = 100
dir_name = 'stories'

documents = []

chars = []
data_list = []
count = []
dictionary = dict({'UNK': 0})
reverse_dictionary = dict()
vocabulary_size = 0

if not os.path.exists(dir_name):
	os.mkdir(dir_name)


def maybe_download(filename):
	"""Download a file if not present"""
	print('Downloading file: ', dir_name + os.sep + filename)
	
	if not os.path.exists(dir_name + os.sep + filename):
		filename, _ = urlretrieve(url + filename, dir_name + os.sep + filename)
	else:
		print('File ', filename, ' already exists.')
	
	return filename


def read_data(filename):
	with open(filename) as f:
		data = tf.compat.as_str(f.read())
		# make all the words lower case  words 转换为小写
		data = data.lower()
		data = list(data)
	return data


def build_dataset(documents):
	'''
	
	:param documents:
	:return:
	count list - [[word,fred]]
	dictionary dict - {word:id}
	reverse_dictionary - {id,word}
	'''
	chars = []
	# This is going to be a list of lists
	# Where the outer list denote each document
	# and the inner lists denote words in a given document
	data_list = []
	
	for d in documents:
		chars.extend(d)
	print('%d Characters found.' % len(chars))
	count = []
	# Get the bigram sorted by their frequency (Highest comes first)
	count.extend(collections.Counter(chars).most_common())
	
	# Create an ID for each bigram by giving the current length of the dictionary
	# And adding that item to the dictionary
	# Start with 'UNK' that is assigned to too rare words
	dictionary = dict({'UNK': 0})
	for char, c in count:
		# Only add a bigram to dictionary if its frequency is more than 10
		if c > 10:
			dictionary[char] = len(dictionary)
	
	unk_count = 0
	# Traverse through all the text we have
	# to replace each string word with the ID of the word
	
	for d in documents:
		data = list()
		for char in d:
			# If word is in the dictionary use the word ID,
			# else use the ID of the special token "UNK"
			if char in dictionary:
				index = dictionary[char]
			else:
				index = dictionary['UNK']
				unk_count += 1
			data.append(index)
		
		data_list.append(data)
	
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data_list, count, dictionary, reverse_dictionary


# 定义样本生成器
class DataGeneratorOHE(object):
	
	def __init__(self, text, batch_size, num_unroll):
		# Text where a bigram is denoted by its ID
		self._text = text
		# Number of bigrams in the text
		self._text_size = len(self._text)
		# Number of datapoints in a batch of data
		self._batch_size = batch_size
		# Num unroll is the number of steps we unroll the RNN in a single training step
		# This relates to the truncated backpropagation we discuss in Chapter 6 text
		self._num_unroll = num_unroll
		# We break the text in to several segments and the batch of data is sampled by
		# sampling a single item from a single segment
		self._segments = self._text_size // self._batch_size
		self._cursor = [offset * self._segments for offset in range(self._batch_size)]
	
	def next_batch(self):
		'''
		Generates a single batch of data
		'''
		# Train inputs (one-hot-encoded) and train outputs (one-hot-encoded)
		batch_data = np.zeros((self._batch_size, vocabulary_size), dtype=np.float32)
		batch_labels = np.zeros((self._batch_size, vocabulary_size), dtype=np.float32)
		
		# Fill in the batch datapoint by datapoint
		for b in range(self._batch_size):
			# If the cursor of a given segment exceeds the segment length
			# we reset the cursor back to the beginning of that segment
			if self._cursor[b] + 1 >= self._text_size:
				self._cursor[b] = b * self._segments
			
			# Add the text at the cursor as the input
			batch_data[b, self._text[self._cursor[b]]] = 1.0
			# Add the preceding bigram as the label to be predicted
			batch_labels[b, self._text[self._cursor[b] + 1]] = 1.0
			# Update the cursor
			self._cursor[b] = (self._cursor[b] + 1) % self._text_size
		
		return batch_data, batch_labels
	
	def unroll_batches(self):
		'''
		This produces a list of num_unroll batches
		as required by a single step of training of the RNN
		'''
		unroll_data, unroll_labels = [], []
		for ui in range(self._num_unroll):
			data, labels = self.next_batch()
			unroll_data.append(data)
			unroll_labels.append(labels)
		
		return unroll_data, unroll_labels
	
	def reset_indices(self):
		'''
		Used to reset all the cursors if needed
		'''
		self._cursor = [offset * self._segments for offset in range(self._batch_size)]


# 定义lstm单元
def lstm_cell(i, o, state, ix, im, ib, cx, cm, cb, fx, fm, fb, ox, om, ob):
	input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
	forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
	update_gate = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
	state = forget_gate * state + input_gate * update_gate
	output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
	return output_gate * tf.tanh(state), state


# 贪婪采样避免单峰
def sample(distribution):
	best_inds = np.argsort(distribution)[-3:]
	best_probs = distribution[best_inds] / np.sum(distribution[best_inds])
	best_idx = np.random.choice(best_inds, p=best_probs)
	return best_idx


# 衰减阀值
decay_threshold = 5
decay_count = 0
min_perplexity = 1e10


def decay_learning_rate(session, v_perplexity, inc_gstep):
	global decay_threshold, decay_count, min_perplexity
	if v_perplexity < min_perplexity:
		decay_count = 0
		min_perplexity = v_perplexity
	else:
		decay_count += 1
	
	if decay_count >= decay_threshold:
		print('\t Reducing learning rate ')
		decay_count = 0
		session.run(inc_gstep)


def main():
	# 下载格林通话前100篇
	filenames = [format(i, '03d') + '.txt' for i in range(1, 101)]
	
	for fn in filenames:
		maybe_download(fn)
	
	# 分词,分成两字符级别,相对于单词可以大大减少维度同样进行字典处理
	global documents
	for i in range(num_files):
		print('\nProcessing file %s' % os.path.join(dir_name, filenames[i]))
		chars = read_data(os.path.join(dir_name, filenames[i]))
		two_grams = [''.join(chars[ch_i:ch_i + 2]) for ch_i in range(0, len(chars) - 2, 2)]
		documents.append(two_grams)
		print('Data size (Characters) (Document %d) %d' % (i, len(two_grams)))
		print('Sample string (Document %d) %s' % (i, two_grams[:50]))
	# 引用全局变量
	global data_list, count, dictionary, reverse_dictionary, vocabulary_size
	data_list, count, dictionary, reverse_dictionary = build_dataset(documents)
	vocabulary_size = len(dictionary)
	
	# 定义超参数
	num_nodes = 128
	
	batch_size = 64
	
	num_unrollings = 50
	
	dropout = 0.2
	
	filename_extension = ''
	if dropout > 0.0:
		filename_extension = '_dropout'
	
	filename_to_save = 'lstm' + filename_extension + '.csv'
	
	tf.reset_default_graph()
	train_inputs, train_labels = [], []
	for ui in range(num_unrollings):
		train_inputs.append(
			tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size], name='train_inputs_%d' % ui))
		train_labels.append(
			tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size], name='train_labels_%d' % ui))
	
	valid_inputs = tf.placeholder(tf.float32, shape=[1, vocabulary_size], name='valid_inputs')
	valid_labels = tf.placeholder(tf.float32, shape=[1, vocabulary_size], name='valid_labels')
	
	test_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size], name='test_input')
	
	# 输入门
	ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], stddev=0.02))
	im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.02))
	ib = tf.Variable(tf.random_uniform([1, num_nodes], -0.02, 0.02))
	
	# 遗忘门
	fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], stddev=0.02))
	fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.02))
	fb = tf.Variable(tf.random_uniform([1, num_nodes], -0.02, 0.02))
	
	# 候选门
	cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], stddev=0.02))
	cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.02))
	cb = tf.Variable(tf.random_uniform([1, num_nodes], -0.02, 0.02))
	
	# 输出门
	ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], stddev=0.02))
	om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.02))
	ob = tf.Variable(tf.random_uniform([1, num_nodes], -0.02, 0.02))
	
	# softmax 参数
	w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], stddev=0.02))
	b = tf.Variable(tf.random_uniform([vocabulary_size], -0.02, 0.02))
	
	# 隐藏状态 & 单元状态
	save_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False, name='train_hidden')
	save_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False, name='train_cell')
	
	save_valid_output = tf.Variable(tf.zeros([1, num_nodes]), trainable=False, name='valid_hidden')
	save_valid_state = tf.Variable(tf.zeros([1, num_nodes]), trainable=False, name='valid_cell')
	
	save_test_output = tf.Variable(tf.zeros([1, num_nodes]), trainable=False, name='test_hidden')
	save_test_state = tf.Variable(tf.zeros([1, num_nodes]), trainable=False, name='test_cell')
	
	outputs = list()
	output = save_output
	state = save_state
	for i in train_inputs:
		output, state = lstm_cell(i, output, state, ix, im, ib, cx, cm, cb, fx, fm, fb, ox, om, ob)
		output = tf.nn.dropout(output, keep_prob=1.0 - dropout)
		outputs.append(output)
	
	logtis = tf.matmul(tf.concat(axis=0, values=outputs), w) + b
	train_prediction = tf.nn.softmax(logits=logtis)
	
	# 计算困惑度
	train_perplexity_without_exp = tf.reduce_sum(
		tf.concat(train_labels, 0) * -tf.log(tf.concat(train_prediction, 0) + 1e-10)) / (num_unrollings * batch_size)
	
	valid_output, valid_state = lstm_cell(valid_inputs, save_valid_output, save_valid_state, ix, im, ib, cx, cm, cb, fx,
	                                      fm, fb, ox, om, ob)
	valid_logtis = tf.nn.xw_plus_b(valid_output, w, b)
	
	with tf.control_dependencies([save_valid_state.assign(valid_state), save_valid_output.assign(valid_output)]):
		valid_prediction = tf.nn.softmax(logits=valid_logtis)
	
	valid_perplexity_without_exp = tf.reduce_sum(valid_labels * -tf.log(valid_prediction + 1e-10))
	
	test_output, test_state = lstm_cell(test_input, save_test_output, save_test_state, ix, im, ib, cx, cm, cb, fx,
	                                    fm, fb, ox, om, ob)
	test_logtis = tf.nn.xw_plus_b(test_output, w, b)
	with tf.control_dependencies([save_test_state.assign(test_state), save_test_output.assign(test_output)]):
		test_predition = tf.nn.softmax(logits=test_logtis)
	
	with tf.control_dependencies([save_output.assign(output), save_state.assign(state)]):
		loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.concat(axis=0, values=train_labels), logits=logtis))
	
	# learning rate decay
	gstep = tf.Variable(0, trainable=False, name='global_step')
	inc_gstep = tf.assign(gstep, gstep + 1)
	
	tf_learning_rate = tf.train.exponential_decay(0.01, gstep, decay_steps=1, decay_rate=0.5)
	
	optimizer = tf.train.AdamOptimizer(tf_learning_rate)
	gradients, v = zip(*optimizer.compute_gradients(loss))
	
	gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
	optimizer = optimizer.apply_gradients(zip(gradients, v))
	
	# Reset train state
	reset_train_state = tf.group(tf.assign(save_state, tf.zeros([batch_size, num_nodes])),
	                             tf.assign(save_output, tf.zeros([batch_size, num_nodes])))
	
	# Reset valid state
	reset_valid_state = tf.group(tf.assign(save_valid_state, tf.zeros([1, num_nodes])),
	                             tf.assign(save_valid_output, tf.zeros([1, num_nodes])))
	
	# Reset test state
	reset_test_state = tf.group(
		save_test_output.assign(tf.random_normal([1, num_nodes], stddev=0.05)),
		save_test_state.assign(tf.random_normal([1, num_nodes], stddev=0.05)))
	
	# 定义全局参数
	# 迭代次数
	num_steps = 26
	step_per_document = 100
	valid_summary = 1
	train_doc_count = 100
	# 每个文本执行的次数
	docs_per_step = 10
	
	train_perplexity_ot = []
	valid_perplexity_ot = []
	
	session = tf.InteractiveSession()
	
	tf.global_variables_initializer().run()
	print('init session')
	
	average_loss = 0
	# 确定10份文件作为验证数据
	long_doc_ids = []
	for di in range(num_files):
		if len(data_list[di]) > 10 * step_per_document:
			long_doc_ids.append(di)
		if len(long_doc_ids) == 10:
			break
	# 生成验证数据
	data_gens = []
	valid_gens = []
	for fi in range(num_files):
		if fi not in long_doc_ids:
			data_gens.append(DataGeneratorOHE(data_list[fi], batch_size, num_unrollings))
		else:
			data_gens.append(DataGeneratorOHE(data_list[fi][:-step_per_document], batch_size, num_unrollings))
			valid_gens.append(DataGeneratorOHE(data_list[fi][-step_per_document:], 1, 1))
	
	feed_dict = {}
	# 代数
	for step in range(num_steps):
		print('Training (Step %d)' % step, end='')
		for di in np.random.permutation(train_doc_count)[:docs_per_step]:
			doc_perplexity = 0
			for doc_step_id in range(step_per_document):
				u_data, u_labels = data_gens[di].unroll_batches()
				
				for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
					feed_dict[train_inputs[ui]] = dat
					feed_dict[train_labels[ui]] = lbl
				
				_, l, step_perplexity = session.run([optimizer, loss, train_perplexity_without_exp],
				                                    feed_dict=feed_dict)
				doc_perplexity += step_perplexity
				
				average_loss += step_perplexity
			print('(%d)' % di, end='')
		print('')
		
		# 生成新的样本
		if (step + 1) % valid_summary == 0:
			average_loss = average_loss / (valid_summary * docs_per_step * step_per_document)
			
			# Print losses
			print('Average loss at step %d: %f' % (step + 1, average_loss))
			print('\tPerplexity at step %d: %f' % (step + 1, np.exp(average_loss)))
			train_perplexity_ot.append(np.exp(average_loss))
			
			average_loss = 0
			valid_loss = 0
			
			for v_doc_id in range(10):
				for v_step in range(step_per_document // 2):
					uvaild_data, uvalid_labels = valid_gens[v_doc_id].unroll_batches()
					v_pred = session.run(valid_perplexity_without_exp,
					                     feed_dict={valid_inputs: uvaild_data[0], valid_labels: uvalid_labels[0]})
					valid_loss += v_pred
				session.run(reset_valid_state)
				valid_gens[v_doc_id].reset_indices()

			v_perplexity = valid_loss / (step_per_document * 10 // 2)
			print("Valid Perplexity: %.2f\n" % v_perplexity)
			valid_perplexity_ot.append(v_perplexity)

			decay_learning_rate(session, v_perplexity,inc_gstep)
			# 开始生成新文本
			segments_to_generate = 1
			chars_in_segment = 500
			
			for _ in range(segments_to_generate):
				print('========================================= New  Text Segment ===================================')
				test_word = np.zeros((1, vocabulary_size), dtype=np.float32)
				# 随机取一个文本
				rand_doc = data_list[np.random.randint(0, num_files)]
				# 随机取一个word
				test_word[0, rand_doc[np.random.randint(0, len(rand_doc))]] = 1.0
				print('\t',reverse_dictionary[np.argmax(test_word[0])],end='')

				for _ in range(chars_in_segment):
					sample_pre = session.run(test_predition,feed_dict={test_input:test_word})
					next_ind = sample(sample_pre.ravel())
					test_word = np.zeros((1, vocabulary_size), dtype=np.float32)
					test_word[0, next_ind] = 1.0
					print(reverse_dictionary[next_ind],end='')
				print()
				session.run(reset_test_state)
				print('===============================================================================================')
			print()
		
	session.close()
					


if __name__ == '__main__':
	main()
