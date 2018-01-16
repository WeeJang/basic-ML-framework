#!/usr/bin/env python2
#-*- coding:utf-8 -*-

import os,sys
from aca import Automaton

class RuleClassifier(object):
	"""
	策略规则引擎
	"""
	def __init__(self,service_type,label_conf_dict_path):
		self.__level1_keywords__ = []
		self.__level1_tag__ = []
		self.__level1_automaton__ = []
		
		label_file = "%s.rule.dat" % (service_type)
		level1_keywords_map_file_path = os.path.join(label_conf_dict_path,label_file)
 
		with open(level1_keywords_map_file_path) as level1_f:
			for line in level1_f:
				if not line:
					continue
				line = line.decode("utf-8").strip()
				if not line:
					continue
				if line.startswith("#"):
					continue
				line_arr = line.split(":")
				keywords_list = line_arr[0].split(" ")
				level1_list = line_arr[1].split(",")

				self.__level1_keywords__.append(keywords_list)
				self.__level1_tag__.append(level1_list)
				automaton = Automaton()
				automaton.add_all(keywords_list)
				self.__level1_automaton__.append(automaton)
	
	def __try_find_tag__(self,content):
		matched_level1_index = []
		for index,automaton in enumerate(self.__level1_automaton__):
			#print "index",index
			match_list = automaton.get_matches(content)
			if len(match_list) == 0:
				continue
			match_str_list = map(lambda elem : elem.elems,match_list)
			#print "content",content
			#print "match_list",
			#for elem in match_str_list:
			#	print elem.encode("utf-8"),
			#print 
			j = 0
			match_str_list_len = len(match_str_list)
			matched_len = 0
			for i,kw in enumerate(self.__level1_keywords__[index]):
				#print "current to match is ",kw.encode("utf-8")
				last_pos = j	
				for x in range(last_pos,match_str_list_len):
					mt = match_str_list[x]
					#print "current to test is ",mt.encode("utf-8")
					j += 1
					if mt == kw:
						matched_len += 1
						#print "yes"
						break
				if j == match_str_list_len:
					break

			if matched_len == len(self.__level1_keywords__[index]):
				#print "this tuple match !"
				matched_level1_index.append(index)
		#print "DEBUG>>>",matched_level1_index		
		return matched_level1_index		

	def test_match(self,content):
		if content is None:
			return None
		tag_index_set = self.__try_find_tag__(content)
		tag_set = set()
		for ind in tag_index_set:
			if ind and ind != -1:
				for tag in self.__level1_tag__[ind]:
					tag_set.add(tag)
		return tag_set,tag_index_set

	def predict_proba(self,raw_data_dict):
		if raw_data_dict is None:
			return None
		heading = u" "
		#heading = raw_data_dict["heading"]
		content = raw_data_dict["content"]
		data = heading + u" " + content
		#print "DEBUG,heading",heading.encode("utf-8")
		tag_index_set = self.__try_find_tag__(data)
		#print "DEBUG,tag_index_set",tag_index_set
		tag_dict = dict()
		for ind in tag_index_set:
			for tag in self.__level1_tag__[ind]:
				counter = tag_dict.get(tag,0) + 1
				tag_dict[tag] = counter
		#print "tag_dict",tag_dict
		
		all_counter = 0
		for tag in tag_dict:
			all_counter += tag_dict[tag]
		
		predict_result = {}
		for tag in tag_dict:
			predict_result[tag] = tag_dict[tag] * 1.0 / all_counter
		return predict_result

	def predict_proba_of_data_list(self,raw_data_dict_list):
		assert isinstance(raw_data_dict_list,list)
		
		predict_proba_list = []
		for raw_data_dict in raw_data_dict_list:
			predict_proba_list.append(self.predict_proba(raw_data_dict))
		return predict_proba_list


