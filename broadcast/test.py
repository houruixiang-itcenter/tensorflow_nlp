#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 下午4:27
# @Author  : Aries
# @Site    :
# @File    : test.py
# @Software: PyCharm
import numpy as np


def main():
	arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
	mean1 = arr.mean(axis=0)
	mean2 = arr.mean(axis=1)
	# print(mean1)
	# print(mean2, mean2.shape)
	
	# 后缘维度 mean1的后缘维度是3  所以会在缺失维度上进行广播,扩展到4 ---> (4,3)
	# print(arr - mean1)
	
	# 后缘纬度不相等,另一个矩阵有一个维度为1,则会在纬度为1上进行广播,扩展到3 ---> (4,3)
	# print(mean2.reshape(4, 1))
	# print(arr - mean2.reshape(4, 1))
	# print(arr - mean2.reshape(4, 1))
	
	# print(mean2.reshape(1, 4))
	# print(arr - mean2.reshape(1, 4))


	# Traceback (most recent call last):
	#   File "/Users/houruixiang/python/tensorflow_nlp/broadcast/test.py", line 30, in <module>
	#     main()
	#   File "/Users/houruixiang/python/tensorflow_nlp/broadcast/test.py", line 26, in main
	#     print(arr - mean2.reshape(1, 4))
	# ValueError: operands could not be broadcast together with shapes (4,3) (1,4)
	
	# a = np.array([[1],[2]])
	# print(a.shape)
	# print(arr - a)
	
	# Traceback (most recent call last):
	# File "/Users/houruixiang/python/tensorflow_nlp/broadcast/test.py", line 41, in <module>
	# File "/Users/houruixiang/python/tensorflow_nlp/broadcast/test.py", line 37, in main
	# print(arr - a)
	# ValueError: operands could not be broadcast together with shapes (4,3) (2,1)
	
	arr1 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
	arr2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
	print(arr1.shape)
	print(arr2.shape)
	print(arr1-arr2)


if __name__ == '__main__':
	main()
