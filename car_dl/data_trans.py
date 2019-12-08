#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 下午4:02
# @Author  : Aries
# @Site    :
# @File    : data_trans.py
# @Software: PyCharm
import pandas as pd
import os
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import time
import matplotlib.pylab as plt
import matplotlib as mlp
from pandas import DataFrame as df
from tensorflow.contrib.layers import batch_norm

import car_dl.pipelineTool as tool
import tensorflow as tf

pd.set_option('display.max_columns', None)


def clean_data(data):
	print('------------------------------------------------check data---------------------------------------------')
	print(data.info())
	print(data.isnull().any())
	# 去除无用的特征
	data_new = data.drop(['行程预估价格', '行程的预估时间', '行程的预估距离', '用户cuid', '订单编号', '订单状态', '用户uid',
	                      '发单时间戳'],
	                     axis=1)
	# 对Nan值做处理
	# data_new.dropna(axis=0, how='any')
	# 对一些超大值列进行限制分档
	data_new['接驾距离'].where(data_new['接驾距离'] < 25428, 25428, inplace=True)
	data_new['接驾时间'].where(data_new['接驾时间'] < 2416, 2416, inplace=True)
	
	# data_fill = tool.fill_data_(data_new)
	data_cleaned = tool.clean_data(data_new)
	data_cleaned = np.array(data_cleaned, dtype=np.float32)
	print('------------------------------------------------cleaning data---------------------------------------------')
	# print(data_cleaned.info())
	# print(data_cleaned.isnull().any())
	# print(data_cleaned)
	# data_set = df(data, columns=data_new.columns)
	return data_cleaned


def main():
	clean_data()


if __name__ == '__main__':
	main()
