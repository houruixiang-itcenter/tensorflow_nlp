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
from tensorflow.contrib import rnn, seq2seq

# import lstm.text_generation.word2vec as word2vec

'''
引入集束搜索
'''

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
class DataGeneratorSeq(object):
	
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
		batch_data = np.zeros((self._batch_size), dtype=np.float32)
		batch_labels = np.zeros((self._batch_size, vocabulary_size), dtype=np.float32)
		
		# Fill in the batch datapoint by datapoint
		for b in range(self._batch_size):
			# If the cursor of a given segment exceeds the segment length
			# we reset the cursor back to the beginning of that segment
			if self._cursor[b] + 1 >= self._text_size:
				self._cursor[b] = b * self._segments
			
			# Add the text at the cursor as the input
			batch_data[b] = self._text[self._cursor[b]]
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


test_word = None
sample_beam_predictions = []
update_sample_beam_state = None
beam_length = 5  # 向前看的步数
beam_neighbors = 5  # 每个时间步的候选项的数量


def get_beam_prediction(session, beam_neighbors, sample_beam_inputs, best_neighbor_beam_indices):
	'''
	
	:param session:
	:param beam_neighbors: 候选集合len
	:param sample_beam_inputs: 集束的输入
	:param best_neighbor_beam_indices: 候选集合
	:return:
	'''
	# Generating words within a segment with Beam Search
	# To make some calculations clearer, we use the example as follows
	# We have three classes with beam_neighbors=2 (best candidate denoted by *, second best candidate denoted by `)
	# For simplicity we assume best candidate always have probability of 0.5 in output prediction
	# second best has 0.2 output prediction
	#           a`                   b*                   c                <--- root level
	#    /     |     \         /     |     \        /     |     \
	#   a      b      c       a*     b`     c      a      b      c         <--- depth 1
	# / | \  / | \  / | \   / | \  / | \  / | \  / | \  / | \  / | \
	# a b c  a b c  a b c   a*b c  a`b c  a b c  a b c  a b c  a b c       <--- depth 2
	# So the best beams at depth 2 would be
	# b-a-a and b-b-a
	global test_word
	global sample_beam_predictions
	global update_sample_beam_state
	feed_dict = {}
	# 更新feed  第一步都是一样的
	for b_n_i in range(beam_neighbors):
		feed_dict.update({sample_beam_inputs[b_n_i]: [test_word]})
	
	# todo 我们计算所有具有相同起始词/字符的邻居的样本预测
	# 这对于更新beam search的所有实例的状态很重要  处理根结点
	sample_preds_root = session.run(sample_beam_predictions, feed_dict=feed_dict)
	# todo 根结点只有一个所以选一个即可 因为初始输出均为一个
	sample_preds_root = sample_preds_root[0]
	
	# indices of top-k candidates
	# b and a in our example (root level)
	# todo 拿到最优的预测概率的索引合集  5个 [::-1]是为了反转  索引其实就是id 因为softmax的输出长度就是整个词汇表的长度
	# todo 返回概率最大的索引
	this_level_candidates = (np.argsort(sample_preds_root, axis=1).ravel()[::-1])[:beam_neighbors].tolist()
	# todo 获取最优序列id对应的softmax概率 5个   --- 注意这里输出的是概率,方便后序输出联合概率
	this_level_prods = sample_preds_root[0, this_level_candidates]
	test_sequences = ['' for _ in range(beam_neighbors)]  # 存储最终的输出
	for b_n_i in range(beam_neighbors):
		# todo 拿到最优概率对应的元素  (2单词的序列)  输出第二层的候选集合
		test_sequences[b_n_i] += reverse_dictionary[this_level_candidates[b_n_i]]
	
	# 计算搜索树的其余深度
	for b_i in range(beam_length - 1):
		test_words = []
		pred_words = []
		
		# 记录每一个集束
		feed_dict = {}
		# todo 遍历最优序列的id
		for p_idx, pred_i in enumerate(this_level_candidates):
			# todo p_idx是索引  pred_i是索引对应的vaule
			# 更新下一个预测feed_dict
			# test_words.append(np.zeros((1, vocabulary_size), dtype=np.float32))
			# test_words[p_idx][0, this_level_candidates[p_idx]] = 1.0
			test_words.append(this_level_candidates[p_idx])
			# todo 更新输入 test_words中记录深度1候选集合的word --- id
			feed_dict.update({sample_beam_inputs[p_idx]: test_words[p_idx]})
		
		# Calculating predictions for all neighbors in beams
		# This is a list of vectors where each vector is the prediction vector for a certain beam
		# For level 1 in our example, the prediction values for
		#      b             a  (previous beam search results)
		# [a,  b,  c],  [a,  b,  c] (current level predictions) would be
		# [0.1,0.1,0.1],[0.5,0.2,0]
		# todo 根据次一级的候选级继续生成预测概率  feed_dict  5个tensor  输出是5个  (1,544)的ndarray
		sample_preds_all_neighbors = session.run(sample_beam_predictions, feed_dict=feed_dict)
		# todo 合并所有次一级候选集合程程的候选集合的概率  此时的输出是(1,544*5)  合并之后索引发生变化,方便求出父节点
		sample_preds_all_neighbors_concat = np.concatenate(sample_preds_all_neighbors, axis=1)
		# 取除最大概率的候选值
		this_level_candidates = np.argsort(sample_preds_all_neighbors_concat.ravel())[::-1][:beam_neighbors]
		
		# todo In the example this would be [1,1] 获取父结点数   5个父节点
		parent_beam_indices = this_level_candidates // vocabulary_size
		# todo 同理计算深度为2的候选集合的5个词id
		this_level_candidates = (this_level_candidates % vocabulary_size).tolist()
		# todo 上一时刻 h, o进行赋值  赋上一个时间点的output&state  因为从depth=2开始 历史的output  并不是一一对应  比如候选1,2可能是同一个output
		session.run(update_sample_beam_state, feed_dict={best_neighbor_beam_indices: parent_beam_indices})
		
		tmp_this_level_probs = np.asarray(this_level_prods)
		tmp_test_sequences = list(test_sequences)
		
		for b_n_i in range(beam_neighbors):
			this_level_prods[b_n_i] = tmp_this_level_probs[parent_beam_indices[b_n_i]]  # 第二层概率
			
			this_level_prods[b_n_i] *= sample_preds_all_neighbors[parent_beam_indices[b_n_i]][
				0, this_level_candidates[b_n_i]]
			
			test_sequences[b_n_i] = tmp_test_sequences[parent_beam_indices[b_n_i]]
			
			test_sequences[b_n_i] += reverse_dictionary[this_level_candidates[b_n_i]]  # todo 父+子
			
			# pred_words.append(np.zeros((1, vocabulary_size), dtype=np.float32))
			# pred_words[b_n_i][0, this_level_candidates[b_n_i]] = 1.0
			pred_words.append(this_level_candidates[b_n_i])
	
	# rand_cand_ids = np.argsort(this_level_prods)[-3:]  # 每一条集束的联合概率排序之后的id
	# rand_cand_prods = this_level_prods[rand_cand_ids] / np.sum(this_level_prods[rand_cand_ids])  # 做一个归一操作?
	# random_id = np.random.choice(rand_cand_ids, p=rand_cand_prods)
	# todo this_level_prods这里记录每一个集束链中的联合概率  parent_beam_indices表示目标预测概率所归属的output序列
	best_beam_id = parent_beam_indices[np.asscalar(np.argmax(this_level_prods))]
	
	session.run(update_sample_beam_state,
	            feed_dict={best_neighbor_beam_indices: [best_beam_id for _ in range(beam_neighbors)]})
	
	test_word = pred_words[best_beam_id]  # todo 重新赋值test_word  下一个时间序列开始的word
	
	return test_sequences[best_beam_id]


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
	
	# 加载word2vec
	embedding_size = 128
	# word2vec.define_data_and_hyperparameters(
	# 	num_files, data_list, reverse_dictionary, embedding_size, vocabulary_size)
	# # word2vec.print_some_batches()
	# word2vec.define_word2vec_tensorflow()
	#
	# # We save the resulting embeddings as embeddings-tmp.npy
	# # If you want to use this embedding for the following steps
	# # please change the name to embeddings.npy and replace the existing
	# word2vec.run_word2vec()
	
	# 定义超参数
	num_nodes = [64, 48, 32]
	
	batch_size = 32
	
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
			tf.placeholder(tf.int32, shape=[batch_size], name='train_inputs_%d' % ui))
		train_labels.append(
			tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size], name='train_labels_%d' % ui))
	
	valid_inputs = tf.placeholder(tf.int32, shape=[1], name='valid_inputs')
	valid_labels = tf.placeholder(tf.float32, shape=[1, vocabulary_size], name='valid_labels')
	
	test_input = tf.placeholder(tf.int32, shape=[1], name='test_input')
	
	
	# todo 加载word2voc嵌入词向量
	embed_mat = np.load('embeddings.npy')
	embed_init = tf.constant(embed_mat)
	embeddings = tf.Variable(embed_init, name='embeddings')
	embedding_size = embed_mat.shape[1]
	
	# 定义真正的输入
	train_inputs_embeds = []
	for ui in range(num_unrollings):
		train_inputs_embeds.append(tf.nn.embedding_lookup(embeddings, train_inputs[ui]))
	
	# Defining embedding lookup for operations for all the validation data
	valid_inputs_embeds = tf.nn.embedding_lookup(embeddings, valid_inputs)
	
	# Defining embedding lookup for operations for all the testing data
	test_input_embeds = tf.nn.embedding_lookup(embeddings, test_input)
	
	# todo 这里不再需要定义lstm的参数,因为我们这里使用tensorflow的API
	# # 输入门
	# ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], stddev=0.02))
	# im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.02))
	# ib = tf.Variable(tf.random_uniform([1, num_nodes], -0.02, 0.02))
	#
	# # 遗忘门
	# fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], stddev=0.02))
	# fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.02))
	# fb = tf.Variable(tf.random_uniform([1, num_nodes], -0.02, 0.02))
	#
	# # 候选门
	# cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], stddev=0.02))
	# cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.02))
	# cb = tf.Variable(tf.random_uniform([1, num_nodes], -0.02, 0.02))
	#
	# # 输出门
	# ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], stddev=0.02))
	# om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=0.02))
	# ob = tf.Variable(tf.random_uniform([1, num_nodes], -0.02, 0.02))
	
	# softmax 参数
	w = tf.Variable(tf.truncated_normal([num_nodes[-1], vocabulary_size], stddev=0.01))
	b = tf.Variable(tf.random_uniform([vocabulary_size], -0.02, 0.02))
	
	# # 隐藏状态 & 单元状态
	# save_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False, name='train_hidden')
	# save_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False, name='train_cell')
	#
	# save_valid_output = tf.Variable(tf.zeros([1, num_nodes]), trainable=False, name='valid_hidden')
	# save_valid_state = tf.Variable(tf.zeros([1, num_nodes]), trainable=False, name='valid_cell')
	#
	# save_test_output = tf.Variable(tf.zeros([1, num_nodes]), trainable=False, name='test_hidden')
	# save_test_state = tf.Variable(tf.zeros([1, num_nodes]), trainable=False, name='test_cell')
	#
	# outputs = list()
	# output = save_output
	# state = save_state
	# for i in train_inputs_embeds:
	# 	output, state = lstm_cell(i, output, state, ix, im, ib, cx, cm, cb, fx, fm, fb, ox, om, ob)
	# 	output = tf.nn.dropout(output, keep_prob=1.0 - dropout)
	# 	outputs.append(output)
	# todo 定义lstm单元列表
	cells = [tf.nn.rnn_cell.LSTMCell(n) for n in num_nodes]
	# 将lstm单元定义DropoutWrapper函数
	'''
	cell:计算中使用的rnn类型
	variational_recurrent:一种特殊的dropout类型
	'''
	dropout_cells = [
		rnn.DropoutWrapper(
			cell=lstm, input_keep_prob=1.0, output_keep_prob=1.0 - dropout, state_keep_prob=1.0,
			variational_recurrent=True, input_size=tf.TensorShape([embedding_size]), dtype=tf.float32
		) for lstm in cells
	]
	
	# todo 接着我们定义一个MultiRNNCell对象,用于封装LSTM单元列表
	stacked_dropout_cell = tf.nn.rnn_cell.MultiRNNCell(dropout_cells)
	stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
	# todo 定义一个张量来进行lstm的迭代的更新状态(隐藏状态和单元状态)
	initial_state = stacked_dropout_cell.zero_state(batch_size, dtype=tf.float32)
	
	# todo 接下来计算lstm单元的输出
	'''
	cell:用于计算输出的序列模型的类型
	inputs:lstm的输入,[num_unrollings, batch_size, embeddings_size],这个shape中时间是0轴,所以这种数据被称为time_major
	time_major:输入的是否为 time major
	initial_state:需要一个起始状态作为起点
	'''
	train_outputs, initial_state = tf.nn.dynamic_rnn(
		stacked_dropout_cell,tf.concat(train_inputs_embeds,axis=0),
		initial_state=initial_state, time_major=True
	)
	
	# todo logtis
	final_output = tf.reshape(train_outputs,[-1,num_nodes[-1]])
	
	
	
	logtis = tf.matmul(final_output, w) + b
	train_prediction = tf.nn.softmax(logits=logtis)
	
	# todo 把logtis转化为time major形式,我们用这种形式来计算损失函数
	time_major_train_logits = tf.reshape(logtis,[num_unrollings,batch_size,-1])
	time_major_train_labels = tf.reshape(tf.concat(train_labels,axis=0),[num_unrollings,batch_size,vocabulary_size])
	
	# 计算困惑度
	train_perplexity_without_exp = tf.reduce_sum(
		tf.concat(train_labels, 0) * -tf.log(tf.concat(train_prediction, 0) + 1e-10)) / (num_unrollings * batch_size)
	
	'''-------------------------------------------------验证集不使用droupout-------------------------------------------------'''
	initial_valid_state = stacked_cell.zero_state(1, dtype=tf.float32)
	
	# Validation input related LSTM computation
	valid_outputs, initial_valid_state = tf.nn.dynamic_rnn(
		stacked_cell, tf.expand_dims(valid_inputs_embeds, 0),
		time_major=True, initial_state=initial_valid_state
	)
	
	final_vaild_output = tf.reshape(valid_outputs, [-1, num_nodes[-1]])
	valid_logtis = tf.nn.xw_plus_b(final_vaild_output, w, b)
	valid_prediction = tf.nn.softmax(logits=valid_logtis)
	
	valid_perplexity_without_exp = tf.reduce_sum(valid_labels * -tf.log(valid_prediction + 1e-10))
	
	# todo 计算loss
	loss = seq2seq.sequence_loss(logtis=tf.transpose(time_major_train_logits, [1, 0, 2]),
	                             targets=tf.transpose(time_major_train_labels),
	                             weights=tf.ones([batch_size, num_unrollings], dtype=tf.float32),
	                             average_across_timesteps=False, average_across_batch=True)
	
	
	loss = tf.reduce_sum(loss)
	

	
	# learning rate decay
	gstep = tf.Variable(0, trainable=False, name='global_step')
	inc_gstep = tf.assign(gstep, gstep + 1)
	
	tf_learning_rate = tf.train.exponential_decay(0.001, gstep, decay_steps=1, decay_rate=0.5)
	
	optimizer = tf.train.AdamOptimizer(tf_learning_rate)
	gradients, v = zip(*optimizer.compute_gradients(loss))
	
	gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
	optimizer = optimizer.apply_gradients(zip(gradients, v))
	
	
	
	'''-----------------------------------------------beam_search----------------------------------------------------'''
	global beam_neighbors
	global beam_length
	# 定义候选步的占位数量,以便在每个时间步保存最佳候选项
	sample_beam_inputs = [tf.placeholder(tf.int32, shape=[1]) for _ in range(beam_neighbors)]
	sample_input = tf.placeholder(tf.int32, shape=[1])
	
	sample_beam_inputs_embeds = [tf.nn.embedding_lookup(embeddings, inp) for inp in sample_beam_inputs]
	sample_input_embeds = tf.nn.embedding_lookup(embeddings, sample_input)
	# todo 定义两个占位符来存放贪婪发现的全局最优集束和本地维护的最佳候选集束索引
	best_beam_index = tf.placeholder(shape=None, dtype=tf.int32)
	# todo 每一个深度的候选集合?
	best_neighbor_beam_indices = tf.placeholder(shape=[beam_neighbors], dtype=tf.int32)
	# todo 为每个集束候选项定义状态和输出变量
	save_sample_beam_output = [tf.Variable(tf.zeros([1, num_nodes])) for _ in range(beam_neighbors)]
	save_sample_beam_state = [tf.Variable(tf.zeros([1, num_nodes])) for _ in range(beam_neighbors)]
	
	# 定义状态重置操作   tf.group就是进行多个操作,返回值不需要关注
	reset_sample_beam_state = tf.group(
		*[save_sample_beam_output[vi].assign(tf.zeros([1, num_nodes])) for vi in range(beam_neighbors)],
		*[save_sample_beam_state[vi].assign(tf.zeros([1, num_nodes])) for vi in range(beam_neighbors)])
	# 我们把它们叠加在下面执行收集操作
	global sample_beam_predictions
	global update_sample_beam_state
	
	# 计算每一个候选项的state and output
	sample_beam_outputs, sample_beam_states = [], []
	tmp_state_tuple = []
	for vi in range(beam_neighbors):
		single_beam_state_tuple = []
		for ni in range(len(num_nodes)):
			single_beam_state_tuple.append(tf.nn.rnn_cell.LSTMStateTuple(save_sample_beam_output[ni][vi],save_sample_beam_state[ni][vi]))
		tmp_state_tuple.append(single_beam_state_tuple)
		
	for vi in range(beam_neighbors):
		sample_beam_outputs.append([])
		sample_beam_states.append([])
		for ni in range(num_nodes):
			sample_beam_outputs[-1].append(tmp_state_tuple[vi][ni][0])
			sample_beam_states[-1].append(tmp_state_tuple[vi][ni][1])
	
	
	stacked_beam_outputs = tf.stack(save_sample_beam_output)  # todo  纵向拼接
	stacked_beam_states = tf.stack(save_sample_beam_state)
	# The beam states for each beam (there are beam_neighbor-many beams) needs to be updated at every depth of tree
	# Consider an example where you have 3 classes where we get the best two neighbors (marked with star)
	#     a`      b*       c
	#   / | \   / | \    / | \
	#  a  b c  a* b` c  a  b  c
	# Since both the candidates from level 2 comes from the parent b
	# We need to update both states/outputs from saved_sample_beam_state/output to have index 1 (corresponding to parent b)
	# todo 赋值上一时刻的  h 和 o; group仅仅是并行的操作集合,返回值无意义
	beam_update_ops = tf.group(
		[[save_sample_beam_output[ni][vi].assign(tf.gather_nd(sample_beam_outputs[vi][ni])) for
		  vi in range(beam_neighbors)]for ni in range(len(num_nodes))],
		[[save_sample_beam_state[ni][vi].assign(tf.gather_nd(sample_beam_states[vi][ni])) for vi
		  in range(beam_neighbors)]for ni in range(len(num_nodes))])
	
	# 1.将上一个time的h and o,以便下次使用
	# 2.计算每一个预测的概率
	for vi in range(beam_neighbors):
		# todo 第一步赋值记忆历史状态  第二步计算概率
		with tf.control_dependencies([beam_update_ops]):
			sample_beam_predictions.append(tf.nn.softmax(tf.nn.xw_plus_b(sample_beam_outputs[vi][-1], w, b)))
	
	filename_to_save = 'lstm_beam_search_dropout'
	
	# Some hyperparameters needed for the training process
	
	num_steps = 26
	steps_per_document = 100
	valid_summary = 1
	train_doc_count = 100
	docs_per_step = 10
	
	beam_nodes = []
	
	beam_train_perplexity_ot = []
	beam_valid_perplexity_ot = []
	session = tf.InteractiveSession()
	
	tf.global_variables_initializer().run()
	
	print('Initialized')
	average_loss = 0
	
	long_doc_ids = []
	for di in range(num_files):
		if len(data_list[di]) > 10 * steps_per_document:
			long_doc_ids.append(di)
		if len(long_doc_ids) == 10:
			break
	
	# Generating validation data
	data_gens = []
	valid_gens = []
	for fi in range(num_files):
		# Get all the bigrams if the document id is not in the validation document ids
		if fi not in long_doc_ids:
			data_gens.append(DataGeneratorSeq(data_list[fi], batch_size, num_unrollings))
		# if the document is in the validation doc ids, only get up to the
		# last steps_per_document bigrams and use the last steps_per_document bigrams as validation data
		else:
			data_gens.append(DataGeneratorSeq(data_list[fi][:-steps_per_document], batch_size, num_unrollings))
			# Defining the validation data generator
			valid_gens.append(DataGeneratorSeq(data_list[fi][-steps_per_document:], 1, 1))
	
	feed_dict = {}
	for step in range(num_steps):
		
		for di in np.random.permutation(train_doc_count)[:docs_per_step]:
			doc_perplexity = 0
			for doc_step_id in range(steps_per_document):
				
				# Get a set of unrolled batches
				u_data, u_labels = data_gens[di].unroll_batches()
				
				# Populate the feed dict by using each of the data batches
				# present in the unrolled data
				for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
					feed_dict[train_inputs[ui]] = dat.reshape(-1).astype(np.int32)
					feed_dict[train_labels[ui]] = lbl
				
				# Running the TensorFlow operations
				_, l, step_perplexity = session.run([optimizer, loss, train_perplexity_without_exp],
				                                    feed_dict=feed_dict)
				# Update doc_perpelxity variable
				doc_perplexity += step_perplexity
				
				# Update the average_loss variable
				average_loss += step_perplexity
			
			# Show the printing progress <train_doc_id_1>.<train_doc_id_2>. ...
			print('(%d).' % di, end='')
		
		# resetting hidden state after processing a single document
		# It's still questionable if this adds value in terms of learning
		# One one hand it's intuitive to reset the state when learning a new document
		# On the other hand this approach creates a bias for the state to be zero
		# We encourage the reader to investigate further the effect of resetting the state
		# session.run(reset_train_state) # resetting hidden state for each document
		print('')
		
		if (step + 1) % valid_summary == 0:
			
			# Compute average loss
			average_loss = average_loss / (docs_per_step * steps_per_document * valid_summary)
			
			# Print loss
			print('Average loss at step %d: %f' % (step + 1, average_loss))
			print('\tPerplexity at step %d: %f' % (step + 1, np.exp(average_loss)))
			beam_train_perplexity_ot.append(np.exp(average_loss))
			
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
						feed_dict={valid_inputs: uvalid_data[0], valid_labels: uvalid_labels[0]}
					)
					
					valid_loss += v_perp
				
				session.run(reset_valid_state)
				
				# Reset validation data generator cursor
				valid_gens[v_doc_id].reset_indices()
			
			print()
			v_perplexity = np.exp(valid_loss / (steps_per_document * 10.0 // 2))
			print("Valid Perplexity: %.2f\n" % v_perplexity)
			beam_valid_perplexity_ot.append(v_perplexity)
			
			# Decay learning rate
			decay_learning_rate(session, v_perplexity, inc_gstep)
			
			# Generating new text ...
			# We will be generating one segment having 500 bigrams
			# Feel free to generate several segments by changing
			# the value of segments_to_generate
			print('Generated Text after epoch %d ... ' % step)
			segments_to_generate = 1
			# 生成500个元素的段落  每次寻魂啊
			chars_in_segment = 500 // beam_length
			
			# todo 从这里开始使用集束搜索来生成文本
			for _ in range(segments_to_generate):
				print('======================== New text Segment ==========================')
				global test_word
				# first word randomly generated
				# 随机选择一个文档进行生成
				rand_doc = data_list[np.random.randint(0, num_files)]
				test_word = rand_doc[np.random.randint(len(rand_doc))]
				print("", reverse_dictionary[test_word], end=' ')
				
				for _ in range(chars_in_segment):
					test_sequence = get_beam_prediction(session, beam_neighbors=beam_neighbors,
					                                    sample_beam_inputs=sample_beam_inputs,
					                                    best_neighbor_beam_indices=best_neighbor_beam_indices)
					print(test_sequence, end='')
				
				print("")
				session.run([reset_sample_beam_state])
				
				print('====================================================================')
			print("")
	
	session.close()
	
	with open(filename_to_save, 'wt') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(beam_train_perplexity_ot)
		writer.writerow(beam_valid_perplexity_ot)


if __name__ == '__main__':
	main()
