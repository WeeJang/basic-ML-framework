#!/usr/bin/env python2
#-*- coding:utf-8 -*-
#from __future__ import unicode_literals

import os,sys

import scipy
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib

from helper_functions import cal_sample_weight,calculate_result,save_data,load_data,get_confusion_matrix,get_classification_report,output_diff_result
from sklearn.model_selection import cross_val_score

class EnsembleClassifier(object):
    """
    基于统计的融合模型(LR + SVM)
    """

    def __init__(self,service_type,conf_dict_path,is_train = False):
        """
        参数:
        - service_type : [before/after]
        - conf_dict_path : 配置文件目录路径
        - is_train : 是否是训练模式,Type boolean,default = False
        """
        self.__conf_dict_path__ = conf_dict_path
        model_name = "%s.ensemble_model.bin" % (service_type)
        self.__model_path__ = os.path.join(self.__conf_dict_path__,model_name)
        # ensemble model
        self.__ensemble_classifier__ = None
        if is_train:
            lr_classifier = LogisticRegression(random_state=1,class_weight="balanced",penalty="l2",C=40)
            svm_classifier = SVC(random_state=1, probability=True, kernel="linear",C=25) 
            self.__ensemble_classifier__ = VotingClassifier(estimators=[("lr",lr_classifier),("svm",svm_classifier)], \
                    voting="soft",weights=[1,0])
        else:
            self.__load_model__()

    def __load_model__(self):
        """
        加载训练好的模型
        """
        self.__ensemble_classifier__ = joblib.load(self.__model_path__)

    def __save_model__(self):
        """
        持久化模型
        """
        joblib.dump(self.__ensemble_classifier__,self.__model_path__)
    
    def enable_proba(self):
        """
        使能返回概率
        """
        self.__ensemble_classifier__.probability = True

    def classes(self):
        """
        返回类别标签
        """
        return self.__ensemble_classifier__.classes_

    def train(self,feature_vec,label_list,**kwargs):
        """
        模型训练接口
        参数:
        - data_list : 原始特征数据list.Type : list of dict
        - label_list : 原始特征数据对应标签list.Type : list of unicode
        返回:
        - Boolean : 训练完成之后，会将模型保存在model_path下。注意，同时FeatureMaker的相关模型参数也会持久在对应的目录下.
        """
        if feature_vec is None:
            return False
        self.__ensemble_classifier__.fit(feature_vec,label_list,sample_weight = None)    
        #scores = cross_val_score(self.__ensemble_classifier__, feature_vec, label_list, cv=5, scoring='accuracy')
        self.__save_model__()
        return True

    def predict(self,feature_vec):
        """
        预测一组结果
        输入:
            feature_vec : 特征向量。Type : dict
        输出：
            预测类别
        """
        if feature_vec is None:
            return 
        # predicts  
        if isinstance(feature_vec,scipy.sparse.csr.csr_matrix):
            feature_vec = feature_vec.toarray()
        return self.__ensemble_classifier__.predict(feature_vec)

    def predict_proba(self,feature_vec):
        """
        预测一条结果
        输入:
            data : 原始特征数据。 Type : dict
        输出：
            预测类别及概率，Type : dict 
            {"label_1":0.4,"label_2":0.4,"label_3":0.2}
        """
        if data is None:
            return 
        if isinstance(feature_vec,scipy.sparse.csr.csr_matrix):
            feature_vec = feature_vec.toarray()
        # predicts    
        labels_list = self.classes().tolist()
        proba_list = self.__ensemble_classifier__.predict_proba(feature_vec).tolist()[0]
        #print "DEBUG,proba_list",proba_list
        return dict(zip(labels_list,proba_list))

