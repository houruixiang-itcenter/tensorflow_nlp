#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 下午8:05
# @Author  : Aries
# @Site    : 
# @File    : word2vec_draw.py
# @Software: PyCharm
import collections
import math
import ssl
import zipfile

import numpy as np
import os
import random
import tensorflow as tf
import bz2
import matplotlib.pyplot as plt
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk  # 标准预处理
import operator  # 按值对字典排序
from math import ceil
import csv
import Word2vec.skip_gram as skip

file = "/Users/houruixiang/python/tensorflow_nlp/Word2vec/dataset/text8.zip"


def find_clustered_embeddings(embeddings, distance_threshold, sample_threshold):
	'''
	Find only the closely clustered embeddings.
	This gets rid of more sparsly distributed word embeddings and make the visualization clearer
	This is useful for t-SNE visualization

	distance_threshold: maximum distance between two points to qualify as neighbors
	sample_threshold: number of neighbors required to be considered a cluster
	'''
	
	# calculate cosine similarity
	cosine_sim = np.dot(embeddings, np.transpose(embeddings))
	norm = np.dot(np.sum(embeddings ** 2, axis=1).reshape(-1, 1),
	              np.sum(np.transpose(embeddings) ** 2, axis=0).reshape(1, -1))
	assert cosine_sim.shape == norm.shape
	cosine_sim /= norm
	
	# make all the diagonal entries zero otherwise this will be picked as highest
	np.fill_diagonal(cosine_sim, -1.0)
	
	argmax_cos_sim = np.argmax(cosine_sim, axis=1)
	mod_cos_sim = cosine_sim
	# find the maximums in a loop to count if there are more than n items above threshold
	for _ in range(sample_threshold - 1):
		argmax_cos_sim = np.argmax(cosine_sim, axis=1)
		mod_cos_sim[np.arange(mod_cos_sim.shape[0]), argmax_cos_sim] = -1
	
	max_cosine_sim = np.max(mod_cos_sim, axis=1)
	
	return np.where(max_cosine_sim > distance_threshold)[0]


def plot(embeddings, labels, title):
	n_clusters = 20  # number of clusters
	# automatically build a discrete set of colors, each for cluster
	# label_colors = [plt.cm.Spectral(float(i) / n_clusters) for i in range(n_clusters)]
	
	assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
	
	# Define K-Means
	kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(embeddings)
	kmeans_labels = kmeans.labels_
	cluster_centroids = kmeans.cluster_centers_
	
	plt.figure(figsize=(15, 15))  # in inches
	plt.title(title)
	# plot all the embeddings and their corresponding words
	for i, (label, klabel) in enumerate(zip(labels, kmeans_labels)):
		# x, y = embeddings[i, :]
		# plt.scatter(x, y)
		#
		# plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
		#              ha='right', va='bottom', fontsize=10)
		
		center = cluster_centroids[klabel, :]
		x, y = embeddings[i, :]
		
		# This is just to spread the data points around a bit
		# So that the labels are clearer
		# We repel datapoints from the cluster centroid
		if x < center[0]:
			x += -abs(np.random.normal(scale=2.0))
		else:
			x += abs(np.random.normal(scale=2.0))
		
		if y < center[1]:
			y += -abs(np.random.normal(scale=2.0))
		else:
			y += abs(np.random.normal(scale=2.0))
		
		plt.scatter(x, y)
		x = x if np.random.random() < 0.5 else x + 10
		y = y if np.random.random() < 0.5 else y - 10
		plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',
		             ha='right', va='bottom', fontsize=16)


def skip_draw():
	# 我们将使用一个大样本空间来建立t-sne流形，然后使用余弦相似性来修剪它
	num_points = 1000
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	words = skip.read_data(file)
	# 创建词典
	data, count, dictionary, reverse_dictionary = skip.build_dataset(words)
	skip_emb_path = os.path.join('..', 'skip_embeddings.npy')
	cbow_emb_path = os.path.join('..', 'cbow_embeddings.npy')
	# 处理skip
	skip_gram_final_embeddings = np.load(skip_emb_path)
	cbow_final_embeddings = np.load(cbow_emb_path)
	
	print('Fitting embeddings to T-SNE. This can take some time ...')
	# get the T-SNE manifold
	skip_gram_selected_embeddings = skip_gram_final_embeddings[:num_points, :]
	skip_gram_two_d_embeddings = tsne.fit_transform(skip_gram_selected_embeddings)
	
	cbow_selected_embeddings = cbow_final_embeddings[:num_points, :]
	cbow_two_d_embeddings = tsne.fit_transform(cbow_selected_embeddings)
	
	print('Pruning the T-SNE embeddings')
	# prune the embeddings by getting ones only more than n-many sample above the similarity threshold
	# this unclutters the visualization
	skip_gram_selected_ids = find_clustered_embeddings(skip_gram_selected_embeddings, .25, 10)
	skip_gram_two_d_embeddings = skip_gram_two_d_embeddings[skip_gram_selected_ids, :]
	print('Out of ', num_points, ' samples, ', skip_gram_selected_ids.shape[0], ' samples were selected by pruning')
	skip_gram_words = [reverse_dictionary[i] for i in skip_gram_selected_ids]
	
	cbow_selected_ids = find_clustered_embeddings(cbow_selected_embeddings, .25, 10)
	cbow_two_d_embeddings = cbow_two_d_embeddings[cbow_selected_ids, :]
	print('Out of ', num_points, ' samples, ', cbow_selected_ids.shape[0], ' samples were selected by pruning')
	cbow_words = [reverse_dictionary[i] for i in cbow_selected_ids]
	
	plot_embeddings_side_by_side(skip_gram_two_d_embeddings, cbow_two_d_embeddings, sg_labels=skip_gram_words,
	                             cbow_labels=cbow_words)


def show_loss_skip_gram_and_cbow():
	cbow_loss_path = os.path.join('..', 'cbow_losses.csv')
	with open(cbow_loss_path, 'rt') as f:
		reader = csv.reader(f, delimiter=',')
		for r_i, row in enumerate(reader):
			if r_i == 0:
				cbow_loss = [float(s) for s in row]
	
	skip_loss_path = os.path.join('..', 'skip_losses.csv')
	with open(skip_loss_path, 'rt') as f:
		reader = csv.reader(f, delimiter=',')
		for r_i, row in enumerate(reader):
			if r_i == 0:
				skip_gram_loss = [float(s) for s in row]
	
	plt.figure(figsize=(15, 5))  # in inches
	
	# Define the x axis
	x = np.arange(len(list(skip_gram_loss))) * 2000
	
	# Plot the skip_gram_loss (loaded from chapter 3)
	plt.plot(x, skip_gram_loss, label="Skip-Gram", linestyle='--', linewidth=2)
	# Plot the cbow_loss (loaded from chapter 3)
	x1 = np.arange(len(list(cbow_loss))) * 2000
	plt.plot(x1, cbow_loss, label="CBOW", linewidth=2)
	
	# Set some text around the plot
	plt.title('Skip-Gram vs CBOW Loss Decrease Over Time', fontsize=24)
	plt.xlabel('Iterations', fontsize=22)
	plt.ylabel('Loss', fontsize=22)
	plt.legend(loc=1, fontsize=22)
	
	# use for saving the figure if needed
	plt.savefig('loss_skipgram_vs_cbow.png')
	plt.show()


def show_distribution_skip_gram_and_cbow():
	skip_emb_path = os.path.join('..', 'skip_embeddings.npy')
	cbow_emb_path = os.path.join('..', 'cbow_embeddings.npy')
	
	skip_gram_final_embeddings = np.load(skip_emb_path)
	cbow_final_embeddings = np.load(cbow_emb_path)
	
	num_points = 1000  # we will use a large sample space to build the T-SNE manifold and then prune it using cosine similarity
	
	# Create a t-SNE object from scikit-learn
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	
	print('Fitting embeddings to T-SNE (skip-gram and CBOW)')
	# Get the T-SNE manifold for skip-gram embeddings
	print('\tSkip-gram')
	sg_selected_embeddings = skip_gram_final_embeddings[:num_points, :]
	sg_two_d_embeddings = tsne.fit_transform(sg_selected_embeddings)
	
	# Get the T-SNE manifold for CBOW embeddings
	print('\tCBOW')
	cbow_selected_embeddings = cbow_final_embeddings[:num_points, :]
	cbow_two_d_embeddings = tsne.fit_transform(cbow_selected_embeddings)
	
	print('Pruning the T-SNE embeddings (skip-gram and CBOW)')
	# Prune the embeddings by getting ones only more than n-many sample above the similarity threshold
	# this unclutters the visualization
	# Prune skip-gram
	print('\tSkip-gram')
	sg_selected_ids = find_clustered_embeddings(sg_selected_embeddings, .3, 10)
	sg_two_d_embeddings = sg_two_d_embeddings[sg_selected_ids, :]
	# Prune CBOW
	print('\tCBOW')
	cbow_selected_ids = find_clustered_embeddings(cbow_selected_embeddings, .3, 10)
	cbow_two_d_embeddings = cbow_two_d_embeddings[cbow_selected_ids, :]
	words = skip.read_data(file)
	
	# 创建词典
	data, count, dictionary, reverse_dictionary = skip.build_dataset(words)
	# Run the function
	sg_words = [reverse_dictionary[i] for i in sg_selected_ids]
	cbow_words = [reverse_dictionary[i] for i in cbow_selected_ids]
	plot_embeddings_side_by_side(sg_two_d_embeddings, cbow_two_d_embeddings, sg_words, cbow_words)


def plot_embeddings_side_by_side(sg_embeddings, cbow_embeddings, sg_labels, cbow_labels):
	''' Plots word embeddings of skip-gram and CBOW side by side as subplots
	'''
	# number of clusters for each word embedding
	# clustering is used to assign different colors as a visual aid
	n_clusters = 20
	
	# automatically build a discrete set of colors, each for cluster
	print('Define Label colors for %d', n_clusters)
	# label_colors = [plt.cm.spectral(float(i) / n_clusters) for i in range(n_clusters)]
	
	# Make sure number of embeddings and their labels are the same
	assert sg_embeddings.shape[0] >= len(sg_labels), 'More labels than embeddings'
	assert cbow_embeddings.shape[0] >= len(cbow_labels), 'More labels than embeddings'
	
	print('Running K-Means for skip-gram')
	# Define K-Means
	sg_kmeans = KMeans(n_clusters=sg_embeddings.shape[0] - 1, init='k-means++', random_state=0).fit(sg_embeddings)
	sg_kmeans_labels = sg_kmeans.labels_
	sg_cluster_centroids = sg_kmeans.cluster_centers_
	
	print('Running K-Means for CBOW')
	cbow_kmeans = KMeans(n_clusters=cbow_embeddings.shape[0] - 1, init='k-means++', random_state=0).fit(cbow_embeddings)
	cbow_kmeans_labels = cbow_kmeans.labels_
	cbow_cluster_centroids = cbow_kmeans.cluster_centers_
	
	print('K-Means ran successfully')
	
	print('Plotting results')
	plt.figure(figsize=(25, 20))  # in inches
	
	# Get the first subplot
	plt.subplot(1, 2, 1)
	
	# Plot all the embeddings and their corresponding words for skip-gram
	for i, (label, klabel) in enumerate(zip(sg_labels, sg_kmeans_labels)):
		center = sg_cluster_centroids[klabel, :]
		x, y = sg_embeddings[i, :]
		
		# This is just to spread the data points around a bit
		# So that the labels are clearer
		# We repel datapoints from the cluster centroid
		if x < center[0]:
			x += -abs(np.random.normal(scale=2.0))
		else:
			x += abs(np.random.normal(scale=2.0))
		
		if y < center[1]:
			y += -abs(np.random.normal(scale=2.0))
		else:
			y += abs(np.random.normal(scale=2.0))
		
		plt.scatter(x, y)
		x = x if np.random.random() < 0.5 else x + 10
		y = y if np.random.random() < 0.5 else y - 10
		plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',
		             ha='right', va='bottom', fontsize=16)
	plt.title('t-SNE for Skip-Gram', fontsize=24)
	
	# Get the second subplot
	plt.subplot(1, 2, 2)
	
	# Plot all the embeddings and their corresponding words for CBOW
	for i, (label, klabel) in enumerate(zip(cbow_labels, cbow_kmeans_labels)):
		center = cbow_cluster_centroids[klabel, :]
		x, y = cbow_embeddings[i, :]
		
		# This is just to spread the data points around a bit
		# So that the labels are clearer
		# We repel datapoints from the cluster centroid
		if x < center[0]:
			x += -abs(np.random.normal(scale=2.0))
		else:
			x += abs(np.random.normal(scale=2.0))
		
		if y < center[1]:
			y += -abs(np.random.normal(scale=2.0))
		else:
			y += abs(np.random.normal(scale=2.0))
		
		plt.scatter(x, y)
		x = x if np.random.random() < 0.5 else x + np.random.randint(0, 10)
		y = y + np.random.randint(0, 5) if np.random.random() < 0.5 else y - np.random.randint(0, 5)
		plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',
		             ha='right', va='bottom', fontsize=16)
	
	plt.title('t-SNE for CBOW', fontsize=24)
	# use for saving the figure if needed
	plt.savefig('tsne_skip_vs_cbow.png')
	plt.show()


def main(Flags):
	skip_draw()


# show_loss_skip_gram_and_cbow()
# show_distribution_skip_gram_and_cbow()


if __name__ == '__main__':
	tf.app.run()
