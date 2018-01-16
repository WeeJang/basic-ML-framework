#!/usr/bin/env python2
#-*- coding:utf-8 -*-

import os,sys

from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.externals import joblib

from FeatureMaker import FeatureMaker
from PersonTypeRecognizer import PersonTypeRecognizer
from helper_functions import calculate_result,save_data,get_confusion_matrix,get_classification_report


class LRClassifier(object):

	def __init__(self,conf_dict_path,model_name):
		"""
		conf_dict_path :  配置文件路径，该路径下包含配置文件，及模型路径
		"""
		model_name = "%s.lr_model.bin" % (model_name)
		self.__model_path__ = os.path.join(conf_dict_path,model_name)
		#self.__lr_clf__ = linear_model.LogisticRegression(C=0.1,penalty = 'l1',tol = 0.01,class_weight="balanced")
		self.__lr_clf__ = linear_model.LogisticRegression(class_weight="balanced")

	def enable_proba(self):
		self.__lr_clf__.probability = True
	
	def train(self,data_list,label_list):
		"""
		data_list : 分词之后的数据["hello world!","hello df!"]
		label_list : 标签list     ["label_1","label_2"]
		"""
		self.enable_proba()
		self.__lr_clf__.fit(data_list,label_list)

	def load_model(self):
		"""
		从硬盘加载模型，字典等
		"""
		self.__lr_clf__ = joblib.load(self.__model_path__)

	def save_model(self):
		"""
		持久化模型，字典等
		"""
		joblib.dump(self.__lr_clf__,self.__model_path__)
		
	def predict(self,data):
		"""
		预测结果
		"""
		return self.__lr_clf__.predict(data)
	
	def classes(self):
		"""
		返回类别标签
		"""
		return self.__lr_clf__.classes_

	def predict_proba(self,data):
		"""
		预测结果,注意只预测一条数据结果#FIXME
		返回:
		{"label_1":0.4,"label_2":0.4,"label_3":0.2}
		"""
		if data is None:
			return None
		labels_list = self.classes().tolist()
		proba_list = self.__lr_clf__.predict_proba(data).tolist()[0]
		#print "debug",labels_list
		#print "debug",proba_list
		return dict(zip(labels_list,proba_list))
		
if __name__ == "__main__":
	import sys,os
	cur_dict_path =os.path.split(os.path.realpath(__file__))[0]
	conf_path = os.path.join(cur_dict_path,"conf")
	
	person_type_recognizer = PersonTypeRecognizer(conf_path)

	def recall_strategy_1(predict,raw_data_list):
		"""
		策略1:针对很多“担保分析”被错分为"财务分析"，增加一个召回策略
		"""
		assert len(predict) == len(raw_data_list)
	
		for index,label in enumerate(predict):
			label = label.strip()
			if label != u"财务分析" \
				and label != u"企业背景" \
				and label != u"经营情况":
				continue
			person_type_set = person_type_recognizer.recognize(raw_data_list[index])
			if person_type_set is None:
				continue
			#print "FUCK!!!"
			if "S" in person_type_set:
				print "YEAH!!!"
				predict[index] = u"担保分析"
		return predict

	def recall_strategy_2(predict,raw_data_list):
		"""
		策略1:针对很多“财务分析”被错分为""，增加一个召回策略
		"""
		assert len(predict) == len(raw_data_list)
	
		for index,label in enumerate(predict):
			label = label.strip()
			if label != u"担保分析":
				continue
			person_type_set = person_type_recognizer.recognize(raw_data_list[index])
			if person_type_set is None:
				continue
			if "S" not in person_type_set:
				predict[index] = u"财务分析"
		return predict


	service_type = sys.argv[1].strip()
	if not(service_type == "after" or \
			service_type == "before"):
		raise Exception("service type is 'after' or 'before'")
	run_type = sys.argv[2].strip()
	if not (run_type == "train" or \
			run_type == "test"):
		raise Exception("run_type is 'train' or 'test'")
	run_type_is_train = True
	if run_type == "test":
		run_type_is_train = False

	# Feature Enginer
	fm = FeatureMaker(conf_path,service_type,is_train = run_type_is_train)
	data_list = []
	label_list = []
	raw_heading_content_data_list = []
	for line in sys.stdin:
		if not line:
			continue
		line_list = line.strip().decode("utf-8").split("<DF_SEP_DF>")
		#print "debug",line_list
		if len(line_list) < 2:
			continue	
		label,heading,content,seg_pos = line_list[0].strip(),line_list[1],line_list[2],line_list[3]
		data = {"heading" : heading,\
				"content" : content,\
				"seg_pos" : float(seg_pos)\
				}
		data_list.append(data)
		label_list.append(label)
		raw_heading_content_data_list.append(heading + " " + content)
	import copy
	data_list = fm.transform(data_list)	
	all_len = len(data_list)
	
	train_len = int(all_len * 0.7)
	train_data,test_data = data_list[:train_len],data_list[train_len:]
	train_label,test_label = label_list[:train_len],label_list[train_len:]	
	"""
	train_len = 0
	train_data,test_data = data_list,data_list
	train_label,test_label = label_list,label_list	
	"""	

	lr_classifier = LRClassifier(conf_path,model_name = service_type)
	if run_type_is_train:
		lr_classifier.train(train_data,train_label)
		lr_classifier.save_model()
	else:
		lr_classifier.load_model()
		predict_result = lr_classifier.predict(test_data)
		predict_result_list = predict_result.tolist()
		predict_result_list = recall_strategy_1(predict_result_list,raw_heading_content_data_list[train_len:])
		#predict_result_list = recall_strategy_2(predict_result_list,raw_data_list[train_len:])
		print "*" * 66 
		calculate_result(test_label,predict_result_list)
		print "*" * 66
		conf_mat = get_confusion_matrix(test_label,predict_result_list)
		print "\n".join(conf_mat).encode("utf-8")
		print 
		print "*" * 66
		report_str = get_classification_report(test_label,predict_result_list)
		print report_str.encode("utf-8")
		
		"""
		print "*" * 66
		for i in range(len(test_label)):
			if test_label[i].strip() == u"担保分析" and \
					predict_result_list[i].strip() == u"财务分析":
				#print test_data[i].encode("utf-8")		
				print raw_data_list[train_len:][i].encode("utf-8")		
		"""
