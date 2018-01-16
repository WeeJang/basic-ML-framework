#!/usr/bin/env python2
#-*- coding:utf-8 -*-

import os,sys

from sklearn.svm import SVC  

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.externals import joblib

from VocabIndexer import VocabIndexer
from helper_functions import calculate_result,save_data
from Cutter import Cutter

class SVMClassifier(object):

	def __init__(self,conf_dict_path,model_name = "before"):
		"""
		conf_dict_path :  配置文件路径，该路径下包含配置文件，及模型路径
		"""
		model_name = "%s.svm_model.pkl" % (model_name)
		self.__model_path__ = os.path.join(conf_dict_path,model_name)
		self.__svm_clf__ = SVC(C = 1.0,kernel='linear',probability=True)

	def enable_proba(self):
		self.__svm_clf__.probability = True
	
	def train(self,data_list,label_list):
		"""
		data_list : 分词之后的数据["hello world!","hello df!"]
		label_list : 标签list     ["label_1","label_2"]
		"""
		self.enable_proba()
		self.__svm_clf__.fit(data_list,label_list)

	def load_model(self):
		"""
		从硬盘加载模型，字典等
		"""
		self.__svm_clf__ = joblib.load(self.__model_path__)

	def save_model(self):
		"""
		持久化模型，字典等
		"""
		joblib.dump(self.__svm_clf__,self.__model_path__)
		
	def predict(self,data):
		"""
		预测结果
		"""
		return self.__svm_clf__.predict(data)
	
	def classes(self):
		"""
		返回类别标签
		"""
		return self.__svm_clf__.classes_

	def predict_proba(self,data):
		"""
		预测结果,注意只预测一条数据结果#FIXME
		返回:
		{"label_1":0.4,"label_2":0.4,"label_3":0.2}
		"""
		if data is None:
			return None
		labels_list = self.classes().tolist()
		proba_list = self.__svm_clf__.predict_proba(data).tolist()[0]
		#print "debug",labels_list
		#print "debug",proba_list
		return dict(zip(labels_list,proba_list))
		
if __name__ == "__main__":
	import sys
	service_type = sys.argv[1].strip()
	if not(service_type == "after" or \
			service_type == "before"):
		raise Exception("service type is 'after' or 'before'")
	conf_path = "/home/jangwee/workspace/wp_py/KP-modularization/src/classifier/conf/"
	voc2ind = VocabIndexer(conf_path)
	voc2ind.load_data()
	cutter = Cutter(conf_path)
	count_vect = CountVectorizer(decode_error='ignore')
	tfidf_transformer = TfidfTransformer()
	
	svm_classifier = SVMClassifier(conf_path,model_name = service_type)
	#svm_classifier.load_model()

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
	#print data_list,len(data_list)
	#split
	all_len = len(data_list)
	train_len = int(all_len * 0.5)
	#train_data,test_data = data_list[:train_len],data_list[train_len:]
	#train_label,test_label = label_list[:train_len],label_list[train_len:]	
	train_data,test_data = data_list,data_list
	train_label,test_label = label_list,label_list	

	train_data_counts = count_vect.fit_transform(train_data)
	train_data_list = tfidf_transformer.fit_transform(train_data_counts)		
	svm_classifier.train(train_data_list,train_label)
	#print count_vect.vocabulary_,type(count_vect.vocabulary_)

	
	count_vect_test = CountVectorizer(decode_error='ignore',vocabulary = count_vect.vocabulary_)
	test_data_counts = count_vect_test.fit_transform(test_data)
	test_data_list = tfidf_transformer.fit_transform(test_data_counts)
	predict_result = svm_classifier.predict(test_data_list)
	#save data {{{
	save_data(count_vect.vocabulary_,"%s/%s.count_vectorizer.vocabulary.pkl" % (conf_path,service_type))
	voc2ind.save_data()
	svm_classifier.save_model()
	# }}}
	predict_result_list = predict_result.tolist()
	calculate_result(test_label,predict_result_list)

