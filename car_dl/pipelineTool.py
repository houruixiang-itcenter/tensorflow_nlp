#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 下午5:24
# @Author  : Aries
# @Site    : 
# @File    : pipelineTool.py
# @Software: PyCharm
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from pandas import DataFrame as df
import numpy as np


def fill_data_(data):
	'''
	get_pipeline_data -- 获取最终数据
	:param data:
	:return:
	'''
	
	num_attrs = list(data)
	fill_pipeline = Pipeline([
		('selector', DataFrameSeletor(num_attrs)),
		('imputer', SimpleImputer(strategy='most_frequent')),
	])
	
	full_pipeline = FeatureUnion(transformer_list=[
		('fill_pipeline', fill_pipeline),
	])
	
	fill_data = df(full_pipeline.fit_transform(data), columns=data.columns)
	return fill_data


def scaler_data(data):
	'''
	get_pipeline_data -- 获取最终数据
	:param data:
	:return:
	'''
	num_attrs = list(data)
	# attrs = ['接驾时间','接驾距离']
	scaler_pipeline = Pipeline([
		('selector', DataFrameSeletor(num_attrs)),
		('std_scaler', StandardScaler()),
	])
	full_pipeline = FeatureUnion(transformer_list=[
		('scaler_pipeline', scaler_pipeline),
	])
	# scalcer_data = df(full_pipeline.fit_transform(data), columns=['接驾时间', '接驾距离'])
	# datatmp = df(full_pipeline.fit_transform(data), columns=['接驾时间', '接驾距离'])
	# data = data.drop(['接驾时间', '接驾距离'], axis=1).append(datatmp, axis=1)
	# data['接驾时间'] = full_pipeline.fit_transform(data)[:,0]
	# data['接驾距离'] = full_pipeline.fit_transform(data)[:,1]
	return full_pipeline.fit_transform(data)


def clean_data(data):
	data_new = data.drop('发单周几', axis=1)
	num_attrs = list(data_new)
	cat_attribs = ['发单周几']
	num_pipeline = Pipeline([
		('selector', DataFrameSeletor(num_attrs)),
		('imputer', SimpleImputer(strategy='most_frequent')),
		('attribs_adder', CombinedAttributeAdder()),
		('std_scaler', StandardScaler()),
	])
	cat_pipeline = Pipeline([
		('selector', DataFrameSeletor(cat_attribs)),
		('label_binarizer', MyLabelBinarizer()),
	])
	full_pipeline = FeatureUnion(transformer_list=[
		('num_pipeline', num_pipeline),
		('cat_pipeline', cat_pipeline),
	])
	return full_pipeline.fit_transform(data)


time, distance, profit = 4, 5, 6


class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room=True):
		self.add_bedrooms_per_room = add_bedrooms_per_room
	
	def fit(self, X, y=None):
		return self  # noting else to do
	
	def transform(self, X, y=None):
		speed = X[:, distance] / X[:, time]
		profit_new = X[:, profit] * 100
		return np.c_[X, speed, profit_new]


class DataFrameSeletor(BaseEstimator, TransformerMixin):
	
	def __init__(self, attrs):
		self.attrs = attrs
	
	def fit(self, X, y=None):
		return self  # noting to do
	
	def transform(self, X, y=None):
		return X[self.attrs].values


class MyLabelBinarizer(TransformerMixin):
	def __init__(self, *args, **kwargs):
		self.encoder = LabelBinarizer(*args, **kwargs)
	
	def fit(self, x, y=0):
		self.encoder.fit(x)
		return self
	
	def transform(self, x, y=None):
		return self.encoder.transform(x)
