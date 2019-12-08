#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 下午2:52
# @Author  : Aries
# @Site    : 
# @File    : data_split.py
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
import car_dl.data_trans as trans

CAR_PATH = '../datasets/'
pd.set_option('display.max_columns', None)


# data_set = df()


def load_car_data(filename):
	csv_path = os.path.join(CAR_PATH, filename)
	return pd.read_csv(csv_path)

def spilt_train_test_vaild():
	train_set, vaild_set, test_set = df(), df(), df()
	for i in range(16, 27):
		data = load_car_data('order_feature_11.%d.csv' % i)
		train_data, test_vaild_data = train_test_split(data, test_size=0.02, random_state=42)
		train_set = train_set.append(train_data)
		test_data, vaild_data = train_test_split(test_vaild_data, test_size=0.5, random_state=42)
		test_set = test_set.append(test_data)
		vaild_set = vaild_set.append(vaild_data)
	
	return train_set, vaild_set, test_set


def get_all_datas_and_labels():
	train_set, vaild_set, test_set = spilt_train_test_vaild()
	train_data = trans.clean_data(train_set.drop(['用户取消真值'], axis=1))
	train_label = train_set['用户取消真值']
	train_label = np.array(train_label).reshape(len(train_label), 1)
	vaild_data = trans.clean_data(vaild_set.drop(['用户取消真值'], axis=1))
	vaild_label = vaild_set['用户取消真值']
	vaild_label = np.array(vaild_label).reshape(len(vaild_label), 1)
	test_data = trans.clean_data(test_set.drop(['用户取消真值'], axis=1))
	test_label = test_set['用户取消真值']
	test_label = np.array(test_label).reshape(len(test_label), 1)
	return train_data, train_label, vaild_data, vaild_label, test_data, test_label


def main():
	print('main')
	data_set = df()
	for i in range(16, 27):
		data = load_car_data('order_feature_11.%d.csv' % i)
		data_set = data_set.append(data_set)
	types = data_set['成单TP号'].value_counts(normalize=True, dropna=False).head()
	print(type)
	

# train_sets, vaild_sets, test_sets = spilt_train_test_vaild()
# print(train_sets.info())
# print('\t')
# print(vaild_sets.info())
# print('\t')
# print(test_sets.info())


if __name__ == '__main__':
	main()
