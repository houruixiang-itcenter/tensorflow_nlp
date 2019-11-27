#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 下午7:05
# @Author  : Aries
# @Site    : 
# @File    : word2vec_improvements.py
# @Software: PyCharm
import collections
import ssl
import zipfile

import numpy as np
import random
import tensorflow as tf
from scipy.sparse import lil_matrix
from six.moves import range
import csv

file = "/Users/houruixiang/python/tensorflow_nlp/Senior_Word2vec/dataset/text8.zip"
# nltk.download('punkt')
vocabulary_size = 50000
data_index = 0


def read_data(file):
	with zipfile.ZipFile(file=file) as f:
		# data = []
		# file_string = f.read(f.namelist()[0]).decode('utf-8')
		# file_string = nltk.word_tokenize(file_string)
		original_data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	# data.extend(file_string)
	return original_data


def build_dataset(words):
	'''
	create a  dict:
	1.maps a string word to an ID & maps an ID to a string word
	2.List of list of (word,frequency)elements(eg.[(I,1),(to,2)...])
	3.Contain the string of text we read,where string words are replaced with word IDs
	:param words:
	:return:
		data: list [[index,id]]   id会重复
		count: list [[word,freq]]
		dictionary: dict [[word,id]]
		reverse_dictionary [[id,word]]
	'''
	count = [['UNK', -1]]
	# collections.Counter(words)计算words中的单词频率
	# .most_common(vocabulary_size - 1)输出排序之后频率最高的5000个词
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	
	# 创建快速查询
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			# dictionary['UNK']
			index = 0
			unk_count = unk_count + 1
		data.append(index)
	
	# update the count variable with the number of UNK occurences
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, window_size, data):
	'''
	generate batch data for skip-gram
	:param batch_size: 批量大小
	:param window_size: 分割窗口大小 就是单词的距离
	:return:
	'''
	# 每次读取数据均要更新数据索引
	global data_index
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	weights = np.ndarray(shape=(batch_size), dtype=np.float32)
	
	# skip window
	span = 2 * window_size + 1
	
	# 缓存区保存span的数据
	buffer = collections.deque(maxlen=span)
	
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	# 需要生成的样本数目
	num_samples = 2 * window_size
	
	for i in range(batch_size // num_samples):
		k = 0
		# 避免使用目标词自身作文模型的上下文预测
		# 填充batch和labels
		for j in list(range(window_size)) + list(range(window_size + 1, 2 * window_size + 1)):
			batch[i * num_samples + k] = buffer[window_size]
			labels[i * num_samples + k, 0] = buffer[j]
			weights[i * num_samples + k] = abs(1.0 / (j - window_size))
			k += 1
		
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels, weights


def main():
	# 读取文件
	words = read_data(file)
	print('len of words is ', len(words))
	print('Example words (start): ', words[:10])
	print('Example words (end): ', words[-10:])
	# 创建词典
	data, count, dictionary, reverse_dictionary = build_dataset(words)
	# batch, labels, weights = generate_batch(batch_size=8, window_size=2, data=data)
	# print(batch.shape, labels.shape, weights.shape)
	
	cooc_data_index = 0
	dataset_size = len(data)
	skip_window = 4
	
	# 存储单词共现的稀疏矩阵 构建稀疏矩阵的一种方式:基于行连接存储的稀疏矩阵
	cooc_mat = lil_matrix((vocabulary_size, vocabulary_size), dtype=np.float32)
	print(cooc_mat.shape)
	
	# def generate_cooc(batch_size, skip_window):
	# 	# Generate co-occurence matrix by processing batches of data
	# 	data_index = 0
	# 	print('Running %d iterations to compute the co-occurance matrix' % (dataset_size // batch_size))
	# 	for i in range(dataset_size // batch_size):
	# 		if i > 0 and i % 100000 == 0:
	# 			print('\tFinished %d iterations' % i)
	#
	# 		# Generating a single batch of data
	# 		batch, labels, weights = generate_batch(batch_size, skip_window, data)
	# 		labels = labels.reshape(-1)
	#
	# 		# 递增稀疏矩阵项
	# 		for inp, lbl, w in zip(batch, labels, weights):
	# 			cooc_mat[inp, lbl] += (1.0 * w)
	#
	# # Generate the matrix
	# generate_cooc(8, skip_window)
	#
	# # Just printing some parts of co-occurance matrix
	# print('Sample chunks of co-occurance matrix')
	#
	# # Basically calculates the highest cooccurance of several chosen word
	# for i in range(10):
	# 	idx_targe = 1
	#
	# 	# get the ith row of the sparse matrix and make it dense
	# 	ith_row = cooc_mat.getrow(idx_targe)
	# 	ith_row_dense = ith_row.toarray('C').reshape(-1)
	#
	# # select target words only with a reasonable words around it.
	# while np.sum(ith_row_dense) < 10 or np.sum(ith_row_dense) > 5000:
	# 	# Choose a random word
	# 	idx_targe = np.random.randint(0, vocabulary_size)
	#
	# 	# get the ith row of the sparse matrix and make it dense
	# 	ith_row = cooc_mat.getrow(idx_targe)
	# 	ith_row_dense = ith_row.toarray('C').reshape(-1)
	#
	# print('\nTarget Word: " %s"' % reverse_dictionary[idx_targe])
	#
	# sort_indices = np.argsort(ith_row_dense).reshape(-1)
	# sort_indices = np.flip(sort_indices, axis=0)
	#
	# print('Context word: ', end='')
	# for j in range(10):
	# 	idx_context = sort_indices[j]
	# 	print('"%s"(id:%d,count:%.2f), ' % (reverse_dictionary[idx_context], idx_context, ith_row_dense[idx_context]),
	# 	      end='')
	# print()
	
	# 开始训练模型
	batch_size = 128
	embeding_size = 128
	window_size = 4
	valid_size = 16
	valid_window = 50
	# 在选择有效的例子时，我们会选择一些最常用的词以及一些比较少见的词
	valid_examples = np.array(random.sample(range(valid_window), valid_size))
	valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)
	# 噪声的数量
	# num_sampled = 32
	
	epsilon = 1
	
	tf.reset_default_graph()
	
	# Training input data (target word IDs).
	train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
	# Training input label data (context word IDs)
	train_labels = tf.placeholder(tf.int64, shape=[batch_size])
	# Validation input data, we don't need a placeholder
	# as we have already defined the IDs of the words selected
	# as validation data
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	
	# 定义模型参数和其他参数
	# 定义Embedding layer
	# I/O嵌入 & I/0偏置嵌入
	in_embeddings = tf.Variable(
		tf.random_uniform([vocabulary_size, embeding_size], -1.0, 1.0), name='embeddings')
	in_bias_embeddings = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01, dtype=tf.float32),
	                                 name='embeddings_bias')
	out_embeddings = tf.Variable(
		tf.random_uniform([vocabulary_size, embeding_size], -1.0, 1.0), name='embeddings')
	out_bias_embeddings = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01, dtype=tf.float32),
	                                  name='embeddings_bias')
	
	embed_in = tf.nn.embedding_lookup(in_embeddings, train_dataset)
	embed_out = tf.nn.embedding_lookup(out_embeddings, train_labels)
	embed_in_bias = tf.nn.embedding_lookup(in_bias_embeddings, train_dataset)
	embed_out_bias = tf.nn.embedding_lookup(out_bias_embeddings, train_labels)
	
	# weights for loss # todo 就是论文中的 f(x_ij)
	weights_x = tf.placeholder(tf.float32, shape=[batch_size], name='weights_x')
	# Cooccurence value for that position
	x_ij = tf.placeholder(tf.float32, shape=[batch_size], name='x_ij')
	
	# 定义loss
	loss = tf.reduce_mean(weights_x * (
			tf.reduce_sum(embed_in * embed_out, axis=1) + embed_in_bias + embed_out_bias - tf.log(
		epsilon + x_ij)) ** 2)
	
	# 计算小批量示例和所有嵌入之间的相似性。我们使用余弦距离：
	norm = tf.sqrt(tf.reduce_sum(tf.square((in_embeddings + out_embeddings) / 2.0), 1, keep_dims=True))
	normalized_embeddings = out_embeddings / norm  # 标准化
	valid_embedings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embedings, tf.transpose(normalized_embeddings))
	
	# 优化器
	optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss, tf.train.get_global_step())
	
	num_step = 100000
	glove_loss = []
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		tf.global_variables_initializer().run()
		average_loss = 0
		for step in range(num_step):
			batch_data, batch_labels, batch_weights = generate_batch(batch_size, window_size, data)
			
			# 计算损失函数需要的weights
			batch_weights = []
			batch_xij = []
			
			# todo  生成f(x_ij)也就是batch_weights 和 x_ij也就是batch_xij
			for inp, lbl in zip(batch_data, batch_labels.reshape(-1)):
				point_weights = (cooc_mat[inp, lbl] / 100) ** 0.75 if cooc_mat[inp, lbl] < 100.0 else 1.0
				batch_weights.append(point_weights)
				batch_xij.append(cooc_mat[inp, lbl])
			batch_weights = np.clip(batch_weights, -100, 1)
			batch_xij = np.asarray(batch_xij)
			
			feed_dict = {train_dataset: batch_data.reshape(-1), train_labels: batch_labels.reshape(-1),

				             weights_x: batch_weights, x_ij: batch_xij}
			
			_, l = sess.run([optimizer, loss], feed_dict=feed_dict)
			
			# print('Average loss at step %d: %f' % (step + 1, l))
			# print(l1)
			average_loss += l
			if (step + 1) % 2000 == 0:
				if step > 0:
					average_loss = average_loss / 500
				glove_loss.append(average_loss)
				print('Average loss at step %d: %f' % (step + 1, average_loss))
				average_loss = 0
			if (step + 1) % 10000 == 0:
				sim = similarity.eval()
				for i in range(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log = 'Nearest to %s:' % valid_word
					for k in range(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log = '%s %s,' % (log, close_word)
					print(log)
		glove_final_embeddings = normalized_embeddings.eval()
	
	# We will save the word vectors learned and the loss over time
	# as this information is required later for comparisons
	# 我们将保存学习到的单词向量和随时间变化的损失，因为稍后需要此信息进行比较
	np.save('./res/glove_embeddings', glove_final_embeddings)
	
	with open('./res/glove_losses.csv', 'wt') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(glove_loss)


if __name__ == '__main__':
	main()
