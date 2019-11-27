#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 上午11:22
# @Author  : Aries
# @Site    : 
# @File    : rnn_language_bigram.py
# @Software: PyCharm
'''RNN-CF 文本生成通话故事'''
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
	
	tf.reset_default_graph()
	num_unroll = 50
	batch_size = 64
	test_batch_size = 1
	
	hidden = 64
	hidden_context = 64
	
	alpha = 0.9
	
	in_size, out_size = vocabulary_size, vocabulary_size
	
	# 初始化  train_dataset,train_labels
	train_dataset, train_labels = [], []
	for ui in range(num_unroll):
		train_dataset.append(tf.placeholder(tf.float32, shape=[batch_size, in_size], name='train_dataset_%d' % ui))
		train_labels.append(tf.placeholder(tf.float32, shape=[batch_size, out_size], name='train_labels_%d' % ui))
	
	# Validation dataset
	valid_dataset = tf.placeholder(tf.float32, shape=[1, in_size], name='valid_dataset')
	valid_labels = tf.placeholder(tf.float32, shape=[1, out_size], name='valid_labels')
	
	# Test dataset
	test_dataset = tf.placeholder(tf.float32, shape=[test_batch_size, in_size], name='save_test_dataset')
	
	A = tf.Variable(tf.truncated_normal([in_size, hidden], stddev=0.02, dtype=tf.float32), name='W_xh')
	B = tf.Variable(tf.truncated_normal([in_size, hidden_context], stddev=0.02, dtype=tf.float32), name='W_xs')
	
	R = tf.Variable(tf.truncated_normal([hidden, hidden], stddev=0.02, dtype=tf.float32), name='W_hh')
	P = tf.Variable(tf.truncated_normal([hidden_context, hidden], stddev=0.02, dtype=tf.float32), name='W_sh')
	
	U = tf.Variable(tf.truncated_normal([hidden, out_size], stddev=0.02, dtype=tf.float32), name='W_hy')
	V = tf.Variable(tf.truncated_normal([hidden_context, out_size], stddev=0.02, dtype=tf.float32), name='W_sy')
	
	prev_train_h = tf.Variable(tf.zeros([batch_size, hidden], dtype=tf.float32), name='train_h', trainable=False)
	prev_train_s = tf.Variable(tf.zeros([batch_size, hidden_context], dtype=tf.float32), name='train_s',
	                           trainable=False)
	
	prev_valid_h = tf.Variable(tf.zeros([1, hidden], dtype=tf.float32), name='valid_h', trainable=False)
	prev_valid_s = tf.Variable(tf.zeros([1, hidden_context], dtype=tf.float32), name='valid_s',
	                           trainable=False)
	
	prev_test_h = tf.Variable(tf.zeros([test_batch_size, hidden], dtype=tf.float32), name='test_h', trainable=False)
	prev_test_s = tf.Variable(tf.zeros([test_batch_size, hidden_context], dtype=tf.float32), name='test_s',
	                          trainable=False)
	
	y_scores, y_predictions = [], []
	next_h_state = prev_train_h
	next_s_state = prev_train_s
	
	# 这里h和s均有输出
	next_h_state_unrolled, next_s_state_unrolled = [], []
	for ui in range(num_unroll):
		next_h_state = tf.nn.tanh(
			tf.matmul(tf.concat([train_dataset[ui], prev_train_h, prev_train_s], 1), tf.concat([A, R, P], 0))
		)
		next_s_state = (1 - alpha) * tf.matmul(train_dataset[ui], B) + alpha * next_s_state
		next_h_state_unrolled.append(next_h_state)
		next_s_state_unrolled.append(next_s_state)
	
	y_scores = [tf.matmul(next_h_state_unrolled[i], U) + tf.matmul(next_s_state_unrolled[i], V) for i in range(num_unroll)]
	y_predictions = [tf.nn.softmax(y_scores[i]) for i in range(num_unroll)]
	
	# 计算困惑度
	train_perplexity_without_exp = tf.reduce_sum(
		tf.concat(train_labels, 0) * -tf.log(tf.concat(y_predictions, 0) + 1e-10)) / (num_unroll * batch_size)
	
	# Compute the next valid state (only for 1 step)
	next_valid_s_state = (1 - alpha) * tf.matmul(valid_dataset, B) + alpha * prev_valid_s
	next_valid_h_state = tf.nn.tanh(tf.matmul(valid_dataset, A) +
	                                tf.matmul(prev_valid_s, P) +
	                                tf.matmul(prev_valid_h, R))
	
	with tf.control_dependencies([tf.assign(prev_valid_s, next_valid_s_state),
	                              tf.assign(prev_valid_h, next_valid_h_state)]):
		valid_scores = tf.matmul(prev_valid_h, U) + tf.matmul(prev_valid_s, V)
		valid_predictions = tf.nn.softmax(valid_scores)
	
	valid_perplexity_without_exp = tf.reduce_sum(valid_labels * -tf.log(valid_predictions + 1e-10))
	
	next_test_s = (1 - alpha) * tf.matmul(test_dataset, B) + alpha * prev_test_s
	
	next_test_h = tf.nn.tanh(
		tf.matmul(test_dataset, A) + tf.matmul(prev_test_s, P) +
		tf.matmul(prev_test_h, R)
	)
	
	with tf.control_dependencies([tf.assign(prev_test_s, next_test_s),
	                              tf.assign(prev_test_h, next_test_h)]):
		test_prediction = tf.nn.softmax(
			tf.matmul(prev_test_h, U) + tf.matmul(prev_test_s, V)
		)
	
	# 计算损失
	with tf.control_dependencies([tf.assign(prev_train_s, next_s_state),
	                              tf.assign(prev_train_h, next_h_state)]):
		rnn_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			logits=tf.concat(y_scores, 0), labels=tf.concat(train_labels, 0)
		))
	
	rnn_valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
		logits=valid_scores, labels=valid_labels))
	
	rnn_optimizer = tf.train.AdamOptimizer(learning_rate=.001)
	
	gradients, v = zip(*rnn_optimizer.compute_gradients(rnn_loss))
	gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
	rnn_optimizer = rnn_optimizer.apply_gradients(zip(gradients, v))
	
	reset_prev_train_h_op = tf.assign(prev_train_h, tf.zeros([batch_size, hidden], dtype=tf.float32))
	reset_prev_train_s_op = tf.assign(prev_train_s, tf.zeros([batch_size, hidden_context], dtype=tf.float32))
	
	reset_valid_h_op = tf.assign(prev_valid_h, tf.zeros([1, hidden], dtype=tf.float32))
	reset_valid_s_op = tf.assign(prev_valid_s, tf.zeros([1, hidden_context], dtype=tf.float32))
	
	# Impute the testing states with noise
	reset_test_h_op = tf.assign(prev_test_h,
	                            tf.truncated_normal([test_batch_size, hidden], stddev=0.01, dtype=tf.float32))
	reset_test_s_op = tf.assign(prev_test_s,
	                            tf.truncated_normal([test_batch_size, hidden_context], stddev=0.01, dtype=tf.float32))
	
	
	# start training
	num_steps = 26
	# todo 在一个步骤中为每个文档执行多少个训练步骤
	steps_per_document = 100
	# 多久执行一次验证步骤
	valid_summary = 1
	
	# todo 我们运行的测试集是20和100
	train_doc_count = 100
	train_docs_to_use = 10
	
	# Store the training and validation perplexity at each step
	cf_valid_perplexity_ot = []
	cf_train_perplexity_ot = []
	
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
		for di in np.random.permutation(train_doc_count)[:train_docs_to_use]:
			doc_perplexity = 0
			for doc_step_id in range(steps_per_document):
				
				# Get a set of unrolled batches
				u_data, u_labels = data_gens[di].unroll_batches()
				
				# Populate the feed dict by using each of the data batches
				# present in the unrolled data
				for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
					feed_dict[train_dataset[ui]] = dat
					feed_dict[train_labels[ui]] = lbl
				
				# Running the TensorFlow operations
				_, l, _, _, _, perp = session.run(
					[rnn_optimizer, rnn_loss, y_predictions, train_dataset,
					 train_labels, train_perplexity_without_exp],
					feed_dict=feed_dict)
				
				# Update doc_perpelxity variable
				doc_perplexity += perp
				
				# Update the average_loss variable
				average_loss += perp
			
			print('Document %d Step %d processed (Perplexity: %.2f).'
			      % (di, step + 1, np.exp(doc_perplexity / (steps_per_document)))
			      )
			
			# resetting hidden state after processing a single document
			# It's still questionable if this adds value in terms of learning
			# One one hand it's intuitive to reset the state when learning a new document
			# On the other hand this approach creates a bias for the state to be zero
			# We encourage the reader to investigate further the effect of resetting the state
			session.run([reset_prev_train_h_op, reset_prev_train_s_op])  # resetting hidden state for each document
		
		# Validation phase
		if step % valid_summary == 0:
			
			# Compute the average validation perplexity
			average_loss = average_loss / (train_docs_to_use * steps_per_document * valid_summary)
			
			# Print losses
			print('Average loss at step %d: %f' % (step + 1, average_loss))
			print('\tPerplexity at step %d: %f' % (step + 1, np.exp(average_loss)))
			
			cf_train_perplexity_ot.append(np.exp(average_loss))
			average_loss = 0  # reset loss
			valid_loss = 0  # reset loss
			
			# calculate valid perplexity
			for v_doc_id in range(10):
				# Remember we process things as bigrams
				# So need to divide by 2
				for v_step in range(steps_per_document // 2):
					uvalid_data, uvalid_labels = valid_gens[v_doc_id].unroll_batches()
					
					# Run validation phase related TensorFlow operations
					v_perp = session.run(
						valid_perplexity_without_exp,
						feed_dict={valid_dataset: uvalid_data[0], valid_labels: uvalid_labels[0]}
					)
					
					valid_loss += v_perp
				
				session.run([reset_valid_h_op, reset_valid_s_op])
				# Reset validation data generator cursor
				valid_gens[v_doc_id].reset_indices()
			
			print()
			v_perplexity = np.exp(valid_loss / (steps_per_document * 10.0 // 2))
			print("Valid Perplexity: %.2f\n" % v_perplexity)
			cf_valid_perplexity_ot.append(v_perplexity)
			
			# Generating new text ...
			# We will be generating one segment having 1000 bigrams
			# Feel free to generate several segments by changing
			# the value of segments_to_generate
			print('Generated Text after epoch %d ... ' % step)
			segments_to_generate = 1
			chars_in_segment = 1000
			
			for _ in range(segments_to_generate):
				print('======================== New text Segment ==========================')
				# Start with a random word
				test_word = np.zeros((1, in_size), dtype=np.float32)
				test_word[0, data_list[np.random.randint(0, num_files)][np.random.randint(0, 100)]] = 1.0
				print("\t", reverse_dictionary[np.argmax(test_word[0])], end='')
				
				# Generating words within a segment by feeding in the previous prediction
				# as the current input in a recursive manner
				for _ in range(chars_in_segment):
					test_pred = session.run(test_prediction, feed_dict={test_dataset: test_word})
					next_ind = sample(test_pred.ravel())
					test_word = np.zeros((1, in_size), dtype=np.float32)
					test_word[0, next_ind] = 1.0
					print(reverse_dictionary[next_ind], end='')
				
				print("")
				# Reset broadcast state
				session.run([reset_test_h_op, reset_test_s_op])
			print("")
	np.save('./res/cf_train_perplexity_ot', cf_train_perplexity_ot)
	np.save('./res/cf_valid_perplexity_ot', cf_valid_perplexity_ot)

	# 处理skip
	# valid_perplexity_ot = np.load('./stories/valid_perplexity_ot')
	# train_perplexity_ot = np.load('./stories/train_perplexity_ot')
	#
	# x_axis = np.arange(len(train_perplexity_ot[1:25]))
	# f, (ax1, ax2) = pylab.subplots(1, 2, figsize=(18, 6))
	#
	# ax1.plot(x_axis, train_perplexity_ot[1:25], label='RNN', linewidth=2, linestyle='--')
	# ax1.plot(x_axis, cf_train_perplexity_ot[1:25], label='RNN-CF', linewidth=2)
	# ax2.plot(x_axis, valid_perplexity_ot[1:25], label='RNN', linewidth=2, linestyle='--')
	# ax2.plot(x_axis, cf_valid_perplexity_ot[1:25], label='RNN-CF', linewidth=2)
	# ax1.legend(loc=1, fontsize=20)
	# ax2.legend(loc=1, fontsize=20)
	# pylab.title('Train and Valid Perplexity over Time (RNN vs RNN-CF)', fontsize=24)
	# ax1.set_title('Train Perplexity', fontsize=20)
	# ax2.set_title('Valid Perplexity', fontsize=20)
	# ax1.set_xlabel('Epoch', fontsize=20)
	# ax2.set_xlabel('Epoch', fontsize=20)
	# pylab.savefig('RNN_perplexity_cf.png')
	# pylab.show()
		
	


if __name__ == '__main__':
	main()
