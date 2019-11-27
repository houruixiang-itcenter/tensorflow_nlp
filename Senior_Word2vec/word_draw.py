#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 下午3:51
# @Author  : Aries
# @Site    : 
# @File    : word_draw.py
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
import os

flags = tf.app.flags
flags.DEFINE_integer("target_func", 0, "Epoch to train [25]")
FLAGS = flags.FLAGS


def draw_original_improved():
	cbow_loss_path = os.path.join('..', 'Word2vec', 'cbow_losses.csv')
	cbow_unigram_loss_path = os.path.join('.', 'cbow_unigram_losses.csv')
	cbow_unigram_sub_loss_path = os.path.join('.', 'cbow_unigram_sub_losses.csv')
	with open(cbow_loss_path, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for index, val in enumerate(reader):
			if index == 0:
				cbow_loss = [float(s) for s in val]
	
	with open(cbow_unigram_loss_path, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for index, val in enumerate(reader):
			if index == 0:
				cbow_unigram_loss = [float(s) for s in val]
	
	with open(cbow_unigram_sub_loss_path, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for index, val in enumerate(reader):
			if index == 0:
				cbow_unigram_sub_loss = [float(s) for s in val]
	
	plt.figure(figsize=(15, 5))
	x = np.arange(len(cbow_loss)) * 2000
	
	# CBOW loss with unigram sampling + subsampling here in one plot
	plt.plot(x, cbow_loss, label="CBOW", linestyle='--', linewidth=2)
	plt.plot(x, cbow_unigram_loss, label="CBOW (Unigram)", linestyle='-.', linewidth=2, marker='^', markersize=5)
	plt.plot(x, cbow_unigram_sub_loss, label="CBOW (Unigram+Subsampling)", linewidth=2)
	
	plt.title('Original vs Improved Skip-Gram Loss Decrease Over Time', fontsize=24)
	plt.xlabel('Iterations', fontsize=22)
	plt.ylabel('Loss', fontsize=22)
	plt.legend(loc=1, fontsize=22)
	
	# use for saving the figure if needed
	plt.savefig('./dataset/loss_cbow.jpg')
	plt.show()


def main(FLAGS):
	draw_original_improved()


if __name__ == '__main__':
	tf.app.run()
