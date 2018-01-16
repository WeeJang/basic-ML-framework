#!/usr/bin/env python2
#-*- coding:utf-8 -*-

import os,sys
from sklearn import decomposition
from sklearn.externals import joblib

from helper_functions import calculate_result,save_data,load_data

class PCAer(object):
	def __init__(self,conf_dict_path,model_name = "before",type_name="heading",dim=160):
		"""
		conf_dict_path :  配置文件路径，该路径下包含配置文件，及模型路径
		"""
		model_name = "%s.%s.pca.bin" % (model_name,type_name)
		self.__model_path__ = os.path.join(conf_dict_path,model_name)
		self.__pca__ = decomposition.TruncatedSVD(n_components = dim, n_iter=1000)
	
	def fit(self,sparse_data_matrix):
		"""
		因为PCA是无监督的，因此不需要送入label
		sparse_data_matrix : 比如tf-idf vecotrized后的稀疏矩阵
		"""
		self.__pca__.fit(sparse_data_matrix)
	
	def transform(self,data):
		"""
		降维
		"""
		return self.__pca__.transform(data)
	
	def fit_transform(self,sparse_data_matrix):
		"""
		"""
		return self.__pca__.fit_transform(sparse_data_matrix)
	def load_model(self):
		"""
		从硬盘加载模型，字典等 """
		self.__pca__ = joblib.load(self.__model_path__)

	def save_model(self):
		"""
		持久化模型，字典等
		"""
		joblib.dump(self.__pca__,self.__model_path__)

	def explained_variance_ratio(self):
		"""
		Percentage of variance explained by each of the selected components.
		"""
		return self.__pca__.explained_variance_ratio_

	def explained_variance(self):
		"""
		The variance of the training samples transformed by a projection to each component.
		"""
		return self.__pca__.explained_variance_


if __name__ == "__main__":
	import sys
	from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
	from VocabIndexer import VocabIndexer
	from Cutter import Cutter
	
	service_type = sys.argv[1].strip()
	if not (service_type == "after" or \
			service_type == "before"):
		raise Exception("service type is 'after' or 'before'")
	
	run_type = sys.argv[2].strip()
	if not (run_type == "train" or \
			run_type == "test"):
		raise Exception("run_type is 'train' or 'test'")

	run_type_is_train = True
	if run_type == "test":
		run_type_is_train = False

	conf_path = "/home/jangwee/workspace/wp_py/KP-modularization/KP_modularization/classifier/conf/"
	voc2ind = VocabIndexer(conf_path)
	voc2ind.load_data()
	cutter = Cutter(conf_path)
	count_vect = None
	count_vec_vocab_path = "%s/%s.count_vectorizer.vocabulary.pkl" % (conf_path,service_type)
	if run_type_is_train:
		count_vect = CountVectorizer(decode_error='ignore')
	else:	
		vocab = load_data(count_vec_vocab_path)
		count_vect = CountVectorizer(decode_error="ignore",vocabulary = vocab)

	tfidf_transformer = TfidfTransformer()
	
	def data2index(data_list):
		ret_list = []
		for elem in data_list:
			ret_list.append(str(voc2ind.voc2id(elem)))
		return ret_list

	data_list = []
	label_list = []
	for line in sys.stdin:
		if not line:
			continue
		line_list = line.strip().decode("utf-8").split("\t")
		#print "debug",line_list
		if len(line_list) < 2:
			continue	
		data,label = line_list[0],line_list[1]
		data = cutter.cut(data)
		data = data2index(data)
		data = " ".join(data)
		data_list.append(data)
		label_list.append(label)		
	
	data_counts = count_vect.fit_transform(data_list)
	if run_type_is_train:
		save_data(count_vect.vocabulary_,count_vec_vocab_path)
	data_list = tfidf_transformer.fit_transform(data_counts)
	pca = PCAer(conf_path,service_type)
	if run_type_is_train:
		pca.fit(data_list)
		pca.save_model()
	else:
		pca.load_model()
		print pca.transform(data_list)
	print "============="	
	vars_ratio_list =  pca.explained_variance_ratio()
	import numpy
	print vars_ratio_list
	print numpy.sum(vars_ratio_list)
