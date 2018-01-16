#!/usr/bin/env python2
#-*- coding:utf-8 -*-
#from __future__ import unicode_literals

import os,sys

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib

from FeatureMaker import FeatureMaker
from helper_functions import calculate_result,save_data,load_data,get_confusion_matrix,get_classification_report,\
                            output_diff_result
from sklearn.model_selection import cross_val_score

from XGBoostClassifier import XGBoostClassifier

def get_classifier(service_type,conf_dict_path,is_train = True):
    """ """
    return XGBoostClassifier(service_type,conf_dict_path,is_train)

def get_feature_maker(service_type,conf_dict_path,is_train = True):
    return  FeatureMaker(conf_dict_path,service_type,is_train)



if __name__ == "__main__":
    import sys,os
    cur_dict_path =os.path.split(os.path.realpath(__file__))[0]
    conf_path = os.path.join(cur_dict_path,"conf")

    service_type = "before"
    if not(service_type == "after" or \
            service_type == "before"):
        raise Exception("service type is 'after' or 'before'")
    
    run_type = sys.argv[1].strip()
    if not (run_type == "train" or \
            run_type == "test"):
        raise Exception("run_type is 'train' or 'test'")
    run_type_is_train = True
    if run_type == "test":
        run_type_is_train = False
   
    classifier = get_classifier(service_type,conf_path,


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
        #label,heading,content,seg_pos = line_list[0].strip(),line_list[1],line_list[2],line_list[3]
        label,heading,content,seg_pos = line_list[0].strip(),u"<NULL>",line_list[1],0
        data = {"heading" : heading,\
                "content" : content,\
                "seg_pos" : float(seg_pos)\
                }
        data_list.append(data)
        label_list.append(label)
        if heading:
            raw_heading_content_data_list.append(heading + u" " + content)
        else:
            raw_heading_content_data_list.append(content)

    all_len = len(data_list)
    #train_len = int(all_len * 0.9)
    train_data,test_data = data_list,data_list
    train_label,test_label = label_list,label_list
    #train_label,test_label = label_list[:train_len],label_list[train_len:]    
    """
    train_len = 0
    train_data,test_data = data_list,data_list
    train_label,test_label = label_list,label_list    
    """
    classifier = None
    if run_type_is_train:
        classifier = get_classifier(service_type,conf_dict_path = conf_path,is_train = True)
        classifier.train(train_data,train_label)
    
    else:
        classifier = get_classifier(service_type,conf_dict_path = conf_path,is_train = False)
        predict_result = classifier.predict(test_data)
        predict_result_list = predict_result.tolist()
        #predict_result_list = recall_strategy_1(predict_result_list,raw_heading_content_data_list[train_len:])
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
        output_diff_result(raw_heading_content_data_list,test_label,predict_result_list,"./result/diff.csv")
