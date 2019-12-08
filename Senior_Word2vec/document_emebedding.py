#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/30 下午5:25
# @Author  : Aries
# @Site    : 
# @File    : document_emebedding.py
# @Software: PyCharm
'''对文档进行分类'''

from __future__ import print_function
import collections
import csv
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk  # standard preprocessing
import operator  # sorting items in dictionary by value
# nltk.download() #tokenizers/punkt/PY3/english.pickle
from math import ceil

file = "/Users/houruixiang/python/tensorflow_nlp_master/Senior_Word2vec/dataset/bbc-fulltext.zip"
# nltk.download('punkt')
vocabulary_size = 25000  # 词汇表中有25000个单词
data_index = 0
test_data_index = 0


def read_data(file):
	data = []
	files_to_read_for_topic = 250
	topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
	with zipfile.ZipFile(file=file) as f:
		parent_dir = f.namelist()[0]
		# 遍历类别
		for t in topics:
			print('\tFinished reading data for topic: ', t)
			# 遍历文档
			for fi in range(1, files_to_read_for_topic):
				with f.open(parent_dir + t + '/' + format(fi, '03d') + '.txt') as f2:
					file_string = f2.read().decode('latin-1')
					file_string = file_string.lower()
					file_string = nltk.word_tokenize(file_string)
					data.extend(file_string)
		# 12250
		return data


def read_test_data(file):
	test_data = {}
	files_to_read_for_topic = 250
	topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
	with zipfile.ZipFile(file=file) as f:
		parent_dir = f.namelist()[0]
		for t in topics:
			print('\tFinished reading data for topic: ', t)
			for fi in np.random.randint(1, files_to_read_for_topic, (10)).tolist():
				with f.open(parent_dir + t + '/' + format(fi, '03d') + '.txt') as f2:
					file_string = f2.read().decode('latin-1')
					file_string = file_string.lower()
					file_string = nltk.word_tokenize(file_string)
					test_data[t + '_' + str(fi)] = file_string
		# 50
		return test_data


def build_dataset(words):
	'''
	create a  dict:
	1.maps a string word to an ID & maps an ID to a string word
	2.List of list of (word,frequency)elements(eg.[(I,1),(to,2)...])
	3.Contain the string of text we read,where string words are replaced with word IDs
	:param words:
	:return:
		data: list [[id]]
		count: list [[word,freq]]
		dictionary: dict [[word,id]]
		reverse_dictionary [[id,word]]
	'''
	count = [['UNK', -1]]
	# collections.Counter(words)计算words中的单词频率
	# .most_common(vocabulary_size - 1)输出排序之后频率最高的25000个词
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))  # 25000
	
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


def build_dataset_with_existing_dictionary(words, dictionary):
	'''
	
	:param words:
	:param dictionary:
	:return:
	'''
	data = list()
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
		data.append(index)
	return data


'''Generating Batches of Data for Skip-Gram'''


def generate_batch(data, batch_size, window_size):
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
	# 初始填充buffer
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	num_samples = 2 * window_size
	# Here we do the batch reading
	# We iterate through each batch index
	# For each batch index, we iterate through span elements
	# to fill in the columns of batch array
	for i in range(batch_size):
		k = 0
		# avoid the target word itself as a prediction
		# fill in batch and label numpy arrays
		for j in range(span):
			if j == span // 2:
				continue
			batch[i, k] = buffer[j]
			k += 1
		labels[i, 0] = buffer[window_size]
		
		# Everytime we read num_samples data points,
		# we have created the maximum number of datapoints possible
		# withing a single span, so we need to move the span by 1
		# to create a fresh new span
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
	return batch, labels


def generate_test_batch(data, batch_size):
	global test_data_index
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	try:
		for bi in range(batch_size):
			batch[bi] = data[test_data_index]
			test_data_index = (test_data_index + 1) % len(data)
	except:
		s = 10
	return batch


def plot(embeddings, labels):
	n_clusters = 5  # number of clusters
	
	# automatically build a discrete set of colors, each for cluster
	# 0-o 1-^ 2-d 3-s 4-x
	label_markers = ['o', '^', 'd', 's', 'x']
	# make sure the number of document embeddings is same as
	# point labels provided
	assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
	
	pylab.figure(figsize=(15, 15))  # in inches
	
	def get_label_id_from_key(key):
		'''
		We assign each different category a cluster_id
		This is assigned based on what is contained in the point label
		Not the actual clustering results
		'''
		if 'business' in key:
			return 0
		elif 'entertainment' in key:
			return 1
		elif 'politics' in key:
			return 2
		elif 'sport' in key:
			return 3
		elif 'tech' in key:
			return 4
	
	# Plot all the document embeddings and their corresponding words
	for i, label in enumerate(labels):
		x, y = embeddings[i, :]
		pylab.scatter(x, y, s=50,
		              marker=label_markers[get_label_id_from_key(label)])
		
		# Annotate each point on the scatter plot
		pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
		               ha='right', va='bottom', fontsize=16)
	
	# Set plot title
	pylab.title('Document Embeddings visualized with t-SNE', fontsize=24)
	
	# Use for saving the figure if needed
	pylab.savefig('./dataset/document_embeddings.png')
	pylab.show()


def main():
	print('Processing training data...')
	words = read_data(file)
	print('\nProcessing testing data...')
	test_words = read_test_data(file)
	
	data, count, dictionary, reverse_dictionary = build_dataset(words)
	test_data = {}
	for k, v in test_words.items():
		print('Building Test Dataset for ', k, ' topic')
		test_data[k] = build_dataset_with_existing_dictionary(test_words[k], dictionary)
	
	batch_size = 128  # Data points in a single batch
	embedding_size = 128  # Dimension of the embedding vector.
	window_size = 4  # How many words to consider left and right.
	
	# We pick a random validation set to sample nearest neighbors
	valid_size = 16  # Random set of words to evaluate similarity on.
	# We sample valid datapoints randomly from a large window without always being deterministic
	valid_window = 50
	
	# When selecting valid examples, we select some of the most frequent words as well as
	# some moderately rare words as well
	# (32,)
	valid_examples = np.array(random.sample(range(valid_window), valid_size))
	valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)
	
	# 负样本
	num_sampled = 32  # Number of negative examples to sample.
	
	tf.reset_default_graph()
	
	# Training input data (target word IDs).
	train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
	
	# Training input label data (context word IDs)
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	
	# Validation input data, we don't need a placeholder
	# as we have already defined the IDs of the words selected
	# as validation data used to evaluate the word vectors
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
	
	# Test data. This is used to compute the document embeddings by averaging
	# word embeddings of a given document
	test_labels = tf.placeholder(tf.int32, shape=[batch_size], name='test_dataset')
	
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))
	softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
	                                                  stddev=1.0 / math.sqrt(embedding_size), dtype=tf.float32))
	softmax_biases = tf.Variable(tf.zeros([vocabulary_size], dtype=tf.float32))
	
	# Used to compute document embeddings by averaging all the word vectors of a
	# given batch of broadcast data
	
	# Used to compute document embeddings by averaging all the word vectors of a
	# given batch of broadcast data
	# 用于计算文档嵌入，方法是平均给定的一批测试数据
	mean_batch_embedding = tf.reduce_mean(tf.nn.embedding_lookup(embeddings, test_labels), axis=0)
	
	# Model.
	# Look up embeddings for all the context words of the inputs.
	# Then compute a tensor by staking embeddings of all context words
	stacked_embedings = None
	print('Defining %d embedding lookups representing each word in the context' % (2 * window_size))
	for i in range(2 * window_size):
		embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
		x_size, y_size = embedding_i.get_shape().as_list()
		if stacked_embedings is None:
			stacked_embedings = tf.reshape(embedding_i, [x_size, y_size, 1])
		else:
			stacked_embedings = tf.concat(axis=2,
			                              values=[stacked_embedings, tf.reshape(embedding_i, [x_size, y_size, 1])])
	
	# Make sure the staked embeddings have 2*window_size columns
	assert stacked_embedings.get_shape().as_list()[2] == 2 * window_size
	print("Stacked embedding size: %s" % stacked_embedings.get_shape().as_list())
	
	# Compute mean embeddings by taking the mean of the tensor containing the stack of embeddings
	mean_embeddings = tf.reduce_mean(stacked_embedings, 2, keepdims=False)
	print("Reduced mean embedding size: %s" % mean_embeddings.get_shape().as_list())
	
	loss = tf.reduce_mean(
		tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
		                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
	
	# Compute the similarity between minibatch examples and all embeddings.
	# We use the cosine distance:
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
	
	# Optimizer. Adagrad optimizers has learning rates assigned to individual parameters
	optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
	
	num_steps = 100001
	cbow_loss = []
	
	with tf.Session() as session:
		
		# Initialize the variables in the graph
		tf.global_variables_initializer().run()
		print('Initialized')
		
		average_loss = 0
		
		# Train the Word2vec model for num_step iterations
		for step in range(num_steps):
			
			# Generate a single batch of data
			batch_data, batch_labels = generate_batch(data, batch_size, window_size)
			
			# Populate the feed_dict and run the optimizer (minimize loss)
			# and compute the loss
			feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
			_, l = session.run([optimizer, loss], feed_dict=feed_dict)
			
			# Update the average loss variable
			average_loss += l
			
			if (step + 1) % 2000 == 0:
				if step > 0:
					average_loss = average_loss / 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print('Average loss at step %d: %f' % (step + 1, average_loss))
				cbow_loss.append(average_loss)
				average_loss = 0
			
			# Evaluating validation set word similarities
			if (step + 1) % 10000 == 0:
				sim = similarity.eval()
				# Here we compute the top_k closest words for a given validation word
				# in terms of the cosine distance
				# We do this for all the words in the validation set
				# Note: This is an expensive step
				for i in range(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log = 'Nearest to %s:' % valid_word
					for k in range(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log = '%s %s,' % (log, close_word)
					print(log)
		final_embedings = normalized_embeddings.eval()
		np.save('./res/document_embedding', final_embedings)
		with open('./res/document_losses.csv', 'wt') as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerow(cbow_loss)
		# Computing broadcast documents embeddings by averaging word embeddings
		
		# We take batch_size*num_test_steps words from each document
		# to compute document embeddings
		num_test_steps = 100
		
		# Store document embeddings
		# {document_id:embedding} format
		document_embeddings = {}
		print('Testing Phase (Compute document embeddings)')
		
		# For each broadcast document compute document embeddings
		for k, v in test_data.items():
			print('\tCalculating mean embedding for document ', k, ' with ', num_test_steps, ' steps.')
			global test_data_index
			test_data_index = 0
			topic_mean_batch_embeddings = np.empty((num_test_steps, embedding_size), dtype=np.float32)
			
			# keep averaging mean word embeddings obtained for each step
			for test_step in range(num_test_steps):
				test_batch_labels = generate_test_batch(test_data[k], batch_size)
				batch_mean = session.run(mean_batch_embedding, feed_dict={test_labels: test_batch_labels})
				topic_mean_batch_embeddings[test_step, :] = batch_mean
			document_embeddings[k] = np.mean(topic_mean_batch_embeddings, axis=0)
		
		
		# Create a t-SNE object
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		
		print('Fitting embeddings to T-SNE')
		# get the T-SNE manifold
		doc_ids, doc_embeddings = zip(*document_embeddings.items())
		two_d_embeddings = tsne.fit_transform(doc_embeddings)
		print('\tDone')
		
		# Run the plotting function
		plot(two_d_embeddings, doc_ids)
		
		print('-------------------------------perform document classification------------------------------')
		kmeans = KMeans(n_clusters=5, random_state=43643, max_iter=10000, n_init=100, algorithm='elkan')
		kmeans.fit(np.array(list(document_embeddings.values())))
		
		# Compute items fallen within each cluster
		document_classes = {}
		for inp, lbl in zip(list(document_embeddings.keys()), kmeans.labels_):
			if lbl not in document_classes:
				document_classes[lbl] = [inp]
			else:
				document_classes[lbl].append(inp)
		for k, v in document_classes.items():
			print('\nDocuments in Cluster ', k)
			print('\t', v)


if __name__ == '__main__':
	main()
