#!/usr/bin/env python2
#-*- coding:utf-8 -*-

import os,sys
import numpy as np
from sklearn.externals import joblib
from xgboost import XGBClassifier

class XGBoostClassifier(object):

    def __init__(self,service_type,conf_dict_path,is_train = True):
        """
        conf_dict_path :  配置文件路径，该路径下包含配置文件，及模型路径
        """
        model_name = "%s.xgb_model.pkl" % (service_type)
        self.__model_path__ = os.path.join(conf_dict_path,model_name)
        self.__xgb_param__ = {"max_depth":3,"learning_rate":0.1,\
                            "objective":"multi:softmax",\
                            "n_estimators":100,\
                            "gamma":0.0,\
                            "base_score":0.5,\
                            "silent":False}
        self.__xgb_clf__ = XGBClassifier(**self.__xgb_param__)

    def enable_proba(self):
        #self.__xgb_clf__.probability = True
        return
    
    def cal_sample_weight(self,label_list):
        """计算每个样本权重
        """
        counter = {}
        for label in label_list:
            counter.setdefault(label,0)
            counter[label] += 1
        mul = 1
        for label in counter:
            mul *= counter[label]
        for label in counter:
            counter[label] = mul / counter[label]
        weight_list = [counter[label] for label in label_list]
        return weight_list 

    def train(self,data_list,label_list,eval_set = None):
        """
        data_list : 分词之后的数据["hello world!","hello df!"]
        label_list : 标签list     ["label_1","label_2"]
        """
        #self.enable_proba()
        sample_weight = self.cal_sample_weight(label_list) 
        self.__xgb_clf__.fit(data_list,label_list,sample_weight,eval_set = eval_set)

    def load_model(self):
        """
        从硬盘加载模型，字典等
        """
        self.__xgb_clf__ = joblib.load(self.__model_path__)

    def save_model(self):
        """
        持久化模型，字典等
        """
        joblib.dump(self.__xgb_clf__,self.__model_path__)
        
    def predict(self,data):
        """
        预测结果
        """
        return self.__xgb_clf__.predict(data)
    
    def classes(self):
        """
        返回类别标签
        """
        return self.__xgb_clf__.classes_

    def predict_proba(self,data):
        """
        预测结果,注意只预测一条数据结果#FIXME
        返回:
        {"label_1":0.4,"label_2":0.4,"label_3":0.2}
        """
        if data is None:
            return None
        labels_list = self.classes().tolist()
        proba_list = self.__xgb_clf__.predict_proba(data.tolist()[0])
        #print "debug",labels_list
        #print "debug",proba_list
        return dict(zip(labels_list,proba_list))
        
