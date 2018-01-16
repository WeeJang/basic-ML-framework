#!/usr/bin/env python2
#-*- coding:utf-8 -*-
import os,sys
import re

import jieba
import jieba.posseg as pseg

class Cutter(object):
	
	def __init__(self,dict_path):
		self.__user_term_path__ = os.path.join(dict_path,"jieba_term.dat")
		jieba.load_userdict(self.__user_term_path__)
		self.__stopwords_path__ = os.path.join(dict_path,"stopwords.dat")
		self.__stopwords_set__ = set()
		self.__stopwords_set__.add(" ")
		with open(self.__stopwords_path__) as f:
			for line in f:
				if not line:
					continue
				elem = line.strip().decode("utf-8")
				self.__stopwords_set__.add(elem)
		#词性过滤
		self.__flag_filter__ = set(["eng","nz","c","a","ad","f","d","e","r","u","b","j","u","uj","i"])

	def cut(self,data):
		if data is None:
			return None
		else:
			assert isinstance(data,unicode),"type(data) %s" % (type(data))
			words_list = pseg.cut(data)
			cutter_list = []
			for elem in words_list:
				word = elem.word
				if elem.flag in self.__flag_filter__:
					continue
				if len(elem.word) == 1:
					continue
				if elem.word in self.__stopwords_set__:
					continue
				
				if elem.flag == "m":
					word = u"<m>"
				if elem.flag == "ns":
					word = u"<ns>"
				if elem.flag == "t":
					word = u"<t>"
				if elem.flag == "nt":
					word = u"<nt>"
				if elem.flag == "mq":
					word = u"<mq>"
				if elem.flag == "nr":
					word = u"<nr>"
				if elem.flag == "q":
					word = u"<q>"
				#for debug
				#cutter_list.append("(" + word.encode("utf-8") + ":" + elem.flag.encode("utf-8")  + ")")
				cutter_list.append(word)
			return cutter_list	

if __name__ == "__main__":
	conf_path = "/home/jangwee/workspace/wp_py/KP-modularization/KP_modularization/classifier/conf"
	jieba_cutter =  Cutter(conf_path)

	for line in sys.stdin:
		if not line:
			continue
		if "standard_tag" in line:
			continue
		line = line.strip().decode("utf-8")
		line_arr = line.split("<DF_SEP_DF>")
		if len(line_arr) < 3:
			continue
		label,heading,content = line_arr[0].strip(),line_arr[1].strip(),line_arr[2].strip()
		all_content = heading + "<EOF>" + content
		seg_list = jieba_cutter.cut(all_content)
		seg_format = " ".join(seg_list)
		print seg_format.encode("utf-8"),"\t",label.encode("utf-8")
