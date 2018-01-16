#!/usr/bin/env python2
#-*- coding:utf-8 -*-

class MixedClassifier(object):
	"""
	MixedClassifier = EnsembleClassifier + RuleClassifier
	               （集成统计模型 + 策略规则引擎)
	"""

	def __init__(self,ensemble_clf,rule_clf):
		"""
		参数:
		- ensemble_clf : 集成统计模型分类器
		- rule_clf  : 策略规则分类器
		"""
		# check if methods implements
		if not ( hasattr(ensemble_clf,"predict_proba") \
				and callable(getattr(ensemble_clf,"predict_proba")) ):
			raise Exception("[FETAL]: ensemble_clf does not have methods 'predict_proba'")
		self.__ensemble_clf__ = ensemble_clf
		
		if not ( hasattr(rule_clf,"predict_proba") \
				and callable(getattr(rule_clf,"predict_proba")) ):
			raise Exception("[FETAL]: rule_clf does not have methods 'predict_proba'")
		self.__rule_clf__ = rule_clf

	def predict_proba(self,data):
		"""
		预测一条结果
		输入:
			data : 原始特征数据。 Type : dict
		输出：
			预测类别及概率，Type : dict 
			{"label_1":0.4,"label_2":0.4,"label_3":0.2}
			#NOTE: 无序!!!
		"""
		if data is None:
			return 
		assert isinstance(data,dict)
	
		# select top_K of ensemble_clf
		# ratio of (top_1 / top_i) < 2
		ensemble_clf_result_top_K = {}
		RATIO_OF_TOP_K = 4.0
		ensemble_clf_result = self.__ensemble_clf__.predict_proba(data)
		#print "DEBUG,ensemble raw result",ensemble_clf_result
		ensemble_clf_result = sorted(ensemble_clf_result.iteritems(),key = lambda entry:entry[1],reverse = True)
		top_1_prob = ensemble_clf_result[0][1]
	
		#print "DEBUG","*" * 10
		#for index,elem in enumerate(ensemble_clf_result):
		#	label,prob = elem[0],elem[1]
		#	print ">>>>>>",label.encode("utf-8"),prob
		#print "DEBUG","*" * 10


		for elem in ensemble_clf_result:
			label,prob = elem[0],elem[1]
			if ( top_1_prob / prob <= RATIO_OF_TOP_K ):
				ensemble_clf_result_top_K[label] = prob
		
		#print "DEBUG,select ensemble raw result",ensemble_clf_result_top_K

		# merge prediction and normolized
		merged_prediction_list = {}
		for elem in ensemble_clf_result_top_K:
			merged_prediction_list[elem] = ensemble_clf_result_top_K[elem]
		
		accum_proba = 0.0
		for elem in merged_prediction_list:
			accum_proba += merged_prediction_list[elem]
		
		for elem in merged_prediction_list:
			merged_prediction_list[elem] = merged_prediction_list[elem] / accum_proba
		#print "DEBUG,merged normalized result",merged_prediction_list
		
		return merged_prediction_list
		
	def predict_proba_by_rule(self,data):
		"""
		预测一条结果
		输入:
			data : 原始特征数据。 Type : dict
		输出：
			预测类别及概率，Type : dict 
			{"label_1":0.4,"label_2":0.4,"label_3":0.2}
			#NOTE: 无序!!!
		"""
		if data is None:
			return 
		assert isinstance(data,dict)
	
		prediction_dict = {}
		# get prediction from rule
		rule_clf_result = self.__rule_clf__.predict_proba(data)
		#print "DEBUG,rule result raw",rule_clf_result

		# merge prediction and normolized
		for elem in rule_clf_result:
			prediction_dict[elem] = rule_clf_result[elem]
		#print "DEBUG,merged raw result",merged_prediction_list
		
		
		return prediction_dict
	
	
