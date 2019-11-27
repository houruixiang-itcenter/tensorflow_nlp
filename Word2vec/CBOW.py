#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 下午4:07
# @Author  : Aries
# @Site    : 
# @File    : CBOW.py
# @Software: PyCharm
import collections
import math
import ssl
import zipfile

import numpy as np
import random
import tensorflow as tf
from six.moves import range
import csv
import Word2vec.utils.word2vec_draw as draw

file = "/Users/houruixiang/python/tensorflow_nlp/Word2vec/dataset/text8.zip"
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
		data: list [[word,id]]
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


def generate_batch_cbow(batch_size, window_size, data):
	# window_size is the amount of words we're looking at from each side of a given word
	# creates a single batch
	
	# data_index is updated by 1 everytime we read a set of data point
	global data_index
	
	# span defines the total window size, where
	# data we consider at an instance looks as follows.
	# [ skip_window target skip_window ]
	# e.g if skip_window = 2 then span = 5
	span = 2 * window_size + 1  # [ skip_window target skip_window ]
	
	# two numpy arras to hold target words (batch)
	# and context words (labels)
	# Note that batch has span-1=2*window_size columns
	batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	
	# The buffer holds the data contained within the span
	buffer = collections.deque(maxlen=span)
	
	# Fill the buffer and update the data_index
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	# Here we do the batch reading
	# We iterate through each batch index
	# For each batch index, we iterate through span elements
	# to fill in the columns of batch array
	for i in range(batch_size):
		target = window_size  # target label at the center of the buffer
		target_to_avoid = [window_size]  # we only need to know the words around a given word, not the word itself
		
		# add selected target to avoid_list for next time
		col_idx = 0
		for j in range(span):
			# ignore the target word when creating the batch
			if j == span // 2:
				continue
			batch[i, col_idx] = buffer[j]
			col_idx += 1
		labels[i, 0] = buffer[target]
		
		# Everytime we read a data point,
		# we need to move the span by 1
		# to create a fresh new span
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	return batch, labels


def main():
	# 读取文件
	words = read_data(file)
	print('len of words is ', len(words))
	print('Example words (start): ', words[:10])
	print('Example words (end): ', words[-10:])
	# 创建词典
	data, count, dictionary, reverse_dictionary = build_dataset(words)
	
	# 开始训练模型
	batch_size = 128
	embeding_size = 128
	window_size = 2
	valid_size = 16
	valid_window = 50
	# 在选择有效的例子时，我们会选择一些最常用的词以及一些比较少见的词
	valid_examples = np.array(random.sample(range(valid_window), valid_size))
	valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)
	# 噪声的数量
	num_sampled = 32
	
	tf.reset_default_graph()
	
	# Training input data (target word IDs).
	train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
	# Training input label data (context word IDs)
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	# Validation input data, we don't need a placeholder
	# as we have already defined the IDs of the words selected
	# as validation data
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	
	# 定义模型参数和其他参数
	# 定义Embedding layer
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embeding_size], -1.0, 1.0, dtype=tf.float32))
	# softmax
	softmax_weights = tf.Variable(
		tf.truncated_normal([vocabulary_size, embeding_size], stddev=0.5 / math.sqrt(embeding_size)))
	softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))
	
	stacked_embedings = None
	print('Defining %d embedding lookups representing each word in the context' % (2 * window_size))
	for i in range(2 * window_size):
		embeding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
		x_size, y_size = embeding_i.get_shape().as_list()
		# 组合cbow(sum)的嵌入词向量层
		if stacked_embedings is None:
			stacked_embedings = tf.reshape(embeding_i, [x_size, y_size, 1])
		else:
			stacked_embedings = tf.concat(axis=2, values=[stacked_embedings, tf.reshape(embeding_i, [x_size, y_size, 1])])
	
	assert stacked_embedings.get_shape().as_list()[2] == 2 * window_size
	print("Stacked embedding size: %s" % stacked_embedings.get_shape().as_list())
	mean_embeddings = tf.reduce_mean(stacked_embedings,2,keepdims=False)
	print("Reduced mean embedding size: %s" % mean_embeddings.get_shape().as_list())
	
	# 定义loss
	loss = tf.reduce_mean(
		tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings, labels=train_labels,
		                           num_sampled=num_sampled, num_classes=vocabulary_size))
	
	# 计算小批量示例和所有嵌入之间的相似性。我们使用余弦距离：
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm  # 标准化
	valid_embedings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) # 筛选单词
	similarity = tf.matmul(valid_embedings, tf.transpose(normalized_embeddings)) # 这里计算的是余弦相似度
	
	# 优化器
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
	
	num_step = 50000
	cbow_loss = []
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		average_loss = 0
		for step in range(num_step):
			# 这里用到python的广播
			batch_data, batch_labels = generate_batch_cbow(batch_size, window_size, data)
			feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
			
			_, l = sess.run([optimizer, loss], feed_dict=feed_dict)
			average_loss += l
			if (step + 1) % 2000 == 0:
				if step > 0:
					average_loss = average_loss / 2000
				cbow_loss.append(average_loss)
				print('Average loss at step %d: %f' % (step + 1, average_loss))
				average_loss = 0
			if (step + 1) % 50000 == 0:
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
		cbow_final_embeddings = normalized_embeddings.eval()
	
	# We will save the word vectors learned and the loss over time
	# as this information is required later for comparisons
	# 我们将保存学习到的单词向量和随时间变化的损失，因为稍后需要此信息进行比较
	np.save('cbow_embeddings', cbow_final_embeddings)
	
	with open('cbow_losses.csv', 'wt') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(cbow_loss)
	
	# 绘制图像
	draw.skip_draw(skip_gram_final_embeddings=cbow_final_embeddings, reverse_dictionary=reverse_dictionary)


if __name__ == '__main__':
	main()
