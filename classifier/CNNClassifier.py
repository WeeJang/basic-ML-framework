#!/usr/bin/env python2
#-*- coding:utf-8 -*-
from __future__ import unicode_literals

import keras
from keras.models import load_model
import pickle
import numpy as np

class CNNClassifier(object):
	
	def __init__(self,model_dict_path,data_type = "before"):
		self.__model__ = None
		self.__word2num__ = None
		self.__label_names = None
		#loading
		if data_type == "before":
			self.__model__ = load_model("%s/cnn_models/cnn_classifier_before.h5" % (model_dict_path,))
			with open("%s/cnn_models/word2num_before.pkl" % (model_dict_path), "rb") as f:
				self.__word2num__ = pickle.load(f)
			with open("%s/cnn_models/label_names_before.pkl" % (model_dict_path), "rb") as g:
				self.__label_names__ = pickle.load(g)
		elif data_type == "after":
			self.__model__ = load_model("%s/cnn_models/cnn_classifier_after.h5" % (model_dict_path))
			with open("%s/cnn_models/word2num_after.pkl" % (model_dict_path), "rb") as f:
				self.__word2num__ = pickle.load(f)
			with open("%s/cnn_models/label_names_after.pkl" % (model_dict_path), "rb") as g:
				self.__label_names__ = pickle.load(g)
		else:
			raise Exception("data_type is before/after\n")

	def predict(self,data):
		#data processing
		text = data.strip()
		#title_content, label = text.split("\t__label__")
		title, content = data.split("EOF")

		title_list = title.split(" ")
		content_list = content.split(" ")

		title2num = [self.__word2num__.get(word, 1) for word in title_list]
		content2num = [self.__word2num__.get(word, 1) for word in content_list]

		if len(title2num) > 20:
			title2num = title2num[:20]
		else:
			title2num = title2num + [0]*(20-len(title2num))
		title_arr = np.array(title2num).reshape((1,20))

		if len(content2num) > 60:
			content2num = content2num[:60]
		else:
			content2num = content2num + [0]*(60-len(content2num))
		content_arr = np.array(content2num).reshape((1,60))
		
		model_input = np.concatenate((title_arr, content_arr), axis=1)
		#model predicting
		pred = self.__model__.predict(model_input)
		
		pred_list = [float(i) for i in pred[0]]
		result = [(self.__label_names__[i], pred_list[i]) for i in range(len(self.__label_names__))]

		return result

if __name__ == "__main__":
	model_dict_path = "/home/jangwee/workspace/wp_py/KP-modularization/src/classifier/conf/"
	classifier = CNNClassifier(model_dict_path)
	#for elem in classifier.predict(u"资信 情况 EOF 申请人 信用 情况 良好"):
	#	print elem[0].encode("utf-8"),elem[1]
