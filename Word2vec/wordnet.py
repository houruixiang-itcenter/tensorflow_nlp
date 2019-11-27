#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 下午4:13
# @Author  : Aries
# @Site    :
# @File    : wordnet.py
# @Software: PyCharm
import nltk
import ssl
from nltk.corpus import wordnet as wn  # 利用nltk下载好wordnet之后开始打导入wordnet语料库
from nltk.corpus.reader import Synset

ssl._create_default_https_context = ssl._create_unverified_context


def main():
	# nltk.download('wordnet')
	# retrieves all the available synsets / 检索所有可用的语法集
	word = 'car'
	car_syns = wn.synsets(word)
	print('retrieves result: ', car_syns)
	# the definition of each synset of car synsets
	syns_def = [car_syns[i].definition() for i in range(len(car_syns))]
	print('synsets definition: \n', syns_def)
	# get the lemmas for the first synset
	car_lemmas = car_syns[0].lemmas()[:3]
	print('the lemmas for the first synset: ', car_lemmas)
	# Let us get hypernyms for a Synset (general superclass) - 上位词
	syn = car_syns[0]  # type:Synset
	print('hypernyms\n', syn.hypernyms()[0].name(), '\n')
	# Let us get hyponyms for a Synset (specific subclass) - 下位词
	print('hyponyms\n', [hypo.name() for hypo in syn.hyponyms()[:3]], '\n')
	# Let us get part-holonyms for the third 'car' Synset (specific subclass)  - 整体词
	print('part-holonyms: ', [i.name() for i in car_syns[2].part_holonyms()], '\n')
	# Let us get meronyms for a Synset (specific subclass)
	print('meronyms: ', [i.name() for i in syn.part_meronyms()[:3]], '\n')


if __name__ == '__main__':
	main()
