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
from six.moves import range
import csv

ssl._create_default_https_context = ssl._create_unverified_context

# 目前下面的链接暂时不能使用 404,not found

# url = 'http://www.evanjones.ca/software/'
#
# def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
#   if not os.path.exists(filename):
#     print('Downloading file...')
#     filename, _ = urlretrieve(url + filename, filename)
#   statinfo = os.stat(filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified %s' % filename)
#   else:
#     print(statinfo.st_size)
#     raise Exception(
#       'Failed to verify ' + filename + '. Can you get to it with a browser?')
#   return filename
#
# filename = maybe_download('wikipedia2text-extracted.txt.bz2', 18377035)
file = "/Users/houruixiang/python/tensorflow_nlp_master/Senior_Word2vec/dataset/text8.zip"
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


def generate_batch_skip_gram(batch_size, window_size, data):
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
			k += 1
		
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
	batch, labels = generate_batch_skip_gram(batch_size=8, window_size=2, data=data)
	print(batch.shape, labels.shape)
	
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
	num_sampled = 32
	
	tf.reset_default_graph()
	
	# Training input data (target word IDs).
	train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
	# Training input label data (context word IDs)
	train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])
	# Validation input data, we don't need a placeholder
	# as we have already defined the IDs of the words selected
	# as validation data
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	
	# 定义模型参数和其他参数
	# 定义Embedding layer
	# todo 在原始的skip-gram论文模型中有I/O
	in_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embeding_size], -1.0, 1.0))
	out_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embeding_size], -1.0, 1.0))
	# todo 对样本进行负采样
	negative_samples, _, _ = tf.nn.log_uniform_candidate_sampler(train_labels, num_true=1, num_sampled=num_sampled,
	                                                             unique=True,
	                                                             range_max=vocabulary_size)
	
	# todo batch转换为词向量的过程 -- 检索嵌入层的过程 这里有输入和输出
	in_embed = tf.nn.embedding_lookup(in_embeddings, train_dataset)
	out_embed = tf.nn.embedding_lookup(out_embeddings, tf.reshape(train_labels, [-1]))
	negative_embed = tf.nn.embedding_lookup(out_embeddings, negative_samples)
	# todo 定义loss
	loss_detail = tf.nn.sigmoid(
		tf.reduce_sum(
			tf.diag([1.0 for _ in range(batch_size)]) *
			tf.matmul(out_embed, tf.transpose(in_embed)),
			axis=0)
	)
	loss1 = tf.reduce_mean(
		tf.log(
			loss_detail
		)
	)
	
	loss = tf.reduce_mean(
		tf.reduce_sum(
			tf.log(tf.nn.sigmoid(-tf.matmul(negative_embed, tf.transpose(in_embed)))),
			axis=0
		)
	)
	
	
	loss = loss + loss1
	
	loss *= -1.0
	
	# 计算小批量示例和所有嵌入之间的相似性。我们使用余弦距离：
	norm = tf.sqrt(tf.reduce_sum(tf.square((in_embeddings + out_embeddings) / 2.0), 1, keep_dims=True))
	normalized_embeddings = out_embeddings / norm  # 标准化
	valid_embedings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embedings, tf.transpose(normalized_embeddings))
	
	# 优化器
	optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss, tf.train.get_global_step())
	
	num_step = 100000
	skip_loss = []
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		tf.global_variables_initializer().run()
		average_loss = 0
		for step in range(num_step):
			# 这里用到python的广播
			batch_data, batch_labels = generate_batch_skip_gram(batch_size, window_size, data)
			feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
			l1 = sess.run(loss_detail, feed_dict=feed_dict)
			_, l = sess.run([optimizer, loss], feed_dict=feed_dict)
			
			# print('Average loss at step %d: %f' % (step + 1, l))
			# print(l1)
			average_loss += l
			if (step + 1) % 2000 == 0:
				if step > 0:
					average_loss = average_loss/500
				skip_loss.append(average_loss)
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
		skip_gram_final_embeddings = normalized_embeddings.eval()
	
	# We will save the word vectors learned and the loss over time
	# as this information is required later for comparisons
	# 我们将保存学习到的单词向量和随时间变化的损失，因为稍后需要此信息进行比较
	np.save('skip_embeddings', skip_gram_final_embeddings)
	
	with open('skip_losses.csv', 'wt') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(skip_loss)
	


if __name__ == '__main__':
	main()
