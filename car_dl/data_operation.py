#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 下午2:52
# @Author  : Aries
# @Site    : 
# @File    : data_operation.py
# @Software: PyCharm
import pandas as pd
import os
import numpy as np
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
import time

CAR_PATH = './datasets/'
pd.set_option('display.max_columns', None)


def load_car_data(car_path=CAR_PATH):
	csv_path = os.path.join(car_path, 'order_feature.csv')
	return pd.read_csv(csv_path)


def test_set_check(identifier, test_radio, hash):
	return hash(np.int64(identifier)).digest()[-1] < 256 * test_radio


def unique_spilt_train_test_by(data, test_radio, id_column, hash=hashlib.md5):
	ids = data[id_column]
	# 测试数据获取 --- from function-test_set_check
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_radio, hash))
	return data.loc[~in_test_set], data.loc[in_test_set]


# 删除没有值的cloumn
def preprocessing_data():
	data = load_car_data()
	data_new = data.drop(['行程预估价格', '行程的预估时间', '行程的预估距离', '用户cuid', '用户cuid'], axis=1)
	return data_new


def spilt_train_test_by_id():
	data = preprocessing_data()
	train_set, test_and_vaild_set = unique_spilt_train_test_by(data, 0.1, '用户uid')
	# ids = len(test_and_vaild_set) * 0.3
	shuffled_indices = np.random.permutation(len(test_and_vaild_set))
	ids = int(len(test_and_vaild_set) * 0.4)
	vaild_indices = shuffled_indices[:ids]
	test_indices = shuffled_indices[ids:]
	# todo 这里的data.iloc就是数据的截断  即20%是测试数据  80%是训练数据
	# todo 需要区分的是  当索引是String类型时候 使用data.loc  反之使用 mldata.iloc
	# todo train_indices/test_indices  是一个索引的数组  通过索引
	return train_set, test_and_vaild_set.iloc[vaild_indices], test_and_vaild_set.iloc[test_indices]


def get_pipeline_data(data):
	'''
	get_pipeline_data -- 获取最终数据
	:param data:
	:return:
	'''
	# num_attrs = list(housing_num)
	# print(num_attrs)
	# cat_attribs = ['ocean_proximity']
	# num_pipeline = Pipeline([
	#     ('selector', DataFrameSeletor(num_attrs)),
	#     ('imputer', SimpleImputer(strategy='median')),
	#     ('attribs_adder', CombinedAttributeAdder()),
	#     ('std_scaler', StandardScaler()),
	# ])
	# cat_pipeline = Pipeline([
	#     ('selector', DataFrameSeletor(cat_attribs)),
	#     ('label_binarizer', MyLabelBinarizer()),
	# ])
	# '''
	# FeatureUnion  Scikit提供 用于整合两个numpy
	# '''
	# print('MyLabelBinarizer :::  ', cat_pipeline.fit_transform(data))
	# full_pipeline = FeatureUnion(transformer_list=[
	#     ('num_pipeline', num_pipeline),
	#     ('cat_pipeline', cat_pipeline),
	# ])
	# pipeline_data = full_pipeline.fit_transform(data)
	# serialize_data(full_pipeline,'full_pipeline')
	# return full_pipeline.fit_transform(data)
	pass


def main():
	train_set, vaild_set, test_set = spilt_train_test_by_id()
	print(train_set.shape)
	print(vaild_set.shape)
	print(test_set.shape)
	tmp = train_set['发单时间戳']
	timeStamp1 = tmp[1]
	localTime1 = time.localtime(timeStamp1)
	strTime1 = time.strftime("%Y-%m-%d %H:%M:%S", localTime1)
	timeStamp2 = tmp[-1]
	localTime2 = time.localtime(timeStamp2)
	strTime2 = time.strftime("%Y-%m-%d %H:%M:%S", localTime2)
	print(strTime1)
	print(strTime2)
	

if __name__ == '__main__':
	main()
