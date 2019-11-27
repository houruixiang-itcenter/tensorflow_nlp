#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/12 下午8:03
# @Author  : Aries
# @Site    : 
# @File    : draw.py
# @Software: PyCharm
import numpy as np
from matplotlib import pylab


def main():
	# 处理skip
	valid_perplexity_ot = np.load('./res/valid_perplexity_ot.npy')
	train_perplexity_ot = np.load('./res/train_perplexity_ot.npy')
	cf_valid_perplexity_ot = np.load('./res/cf_valid_perplexity_ot.npy')
	cf_train_perplexity_ot = np.load('./res/cf_train_perplexity_ot.npy')
	
	x_axis = np.arange(len(train_perplexity_ot[1:25]))
	f, (ax1, ax2) = pylab.subplots(1, 2, figsize=(18, 6))
	
	ax1.plot(x_axis, train_perplexity_ot[1:25], label='RNN', linewidth=2, linestyle='--')
	ax1.plot(x_axis, cf_train_perplexity_ot[1:25], label='RNN-CF', linewidth=2)
	ax2.plot(x_axis, valid_perplexity_ot[1:25], label='RNN', linewidth=2, linestyle='--')
	ax2.plot(x_axis, cf_valid_perplexity_ot[1:25], label='RNN-CF', linewidth=2)
	ax1.legend(loc=1, fontsize=20)
	ax2.legend(loc=1, fontsize=20)
	pylab.title('Train and Valid Perplexity over Time (RNN vs RNN-CF)', fontsize=24)
	ax1.set_title('Train Perplexity', fontsize=20)
	ax2.set_title('Valid Perplexity', fontsize=20)
	ax1.set_xlabel('Epoch', fontsize=20)
	ax2.set_xlabel('Epoch', fontsize=20)
	pylab.savefig('RNN_perplexity_cf.png')
	pylab.show()


if __name__ == '__main__':
	main()
