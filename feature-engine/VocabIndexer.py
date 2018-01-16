#!/usr/bin/env python2
#-*- coding:utf-8 -*-

import cPickle

class VocabIndexer(object):

	def __init__(self,bin_dict_path,dict_name):
		"""
		bin_dict_path :  字典文件路径，该路径下包含字典文件
		"""
		self.__bin_dict_path__ = bin_dict_path
		self.__dict_name__ = dict_name
		self.__voc2id__ = dict()
		self.__id2voc__ = dict()

	def try_voc2id(self,vocab):
		return self.__voc2id__.get(vocab)

	def voc2id(self,vocab):
		"""
		返回index
		参数：
			vocab : unicode type,单词
		返回：
			index : int type,索引
		"""
		target_id = self.__voc2id__.get(vocab)
		if target_id is None:
			cur_len = len(self.__voc2id__)
			self.__voc2id__[vocab] = cur_len
			self.__id2voc__[cur_len] = vocab
			target_id = cur_len
		return target_id

	def id2voc(self,vocab):
		"""
		返回index
		参数：
			index : int type,索引
		返回：
			vocab : unicode type,单词
		"""
		target_id = self.__id2voc__.get(vocab)
		return target_id

	def load_data(self):
		"""
		从硬盘加载模型，字典等
		"""
		voc2id_p = "%s/%s.voc2id.pkl" % (self.__bin_dict_path__,self.__dict_name__)
		id2voc_p = "%s/%s.id2voc.pkl" % (self.__bin_dict_path__,self.__dict_name__)
		with open(voc2id_p, 'rb') as fid:
		    self.__voc2id__ = cPickle.load(fid)
		with open(id2voc_p, 'rb') as fid:
		    self.__id2voc__ = cPickle.load(fid)
		return True

	def save_data(self):
		"""
		持久化模型，字典等
		"""
		voc2id_p = "%s/%s.voc2id.pkl" % (self.__bin_dict_path__,self.__dict_name__)
		id2voc_p = "%s/%s.id2voc.pkl" % (self.__bin_dict_path__,self.__dict_name__)
		with open(voc2id_p, 'wb') as fid:
		    cPickle.dump(self.__voc2id__, fid)
		with open(id2voc_p, 'wb') as fid:
		    cPickle.dump(self.__id2voc__, fid)
		return True
	
