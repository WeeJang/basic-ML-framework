#!/usr/bin/env python2
#-*- coding:utf-8 -*-

import cPickle
import collections
import logging

from sklearn import metrics

logging.basicConfig()
logger = logging.getLogger()
#logger.setLevel("DEBUG")

def cal_sample_weight(label_list):
    """
    根据balanced原则，计算样本权重
    - label_list
    """
    all_length = len(label_list)
    counter = collections.Counter()
    for elem in label_list:
        counter[elem] += 1
    weight_dict = {}
    accum = 0.0
    for elem in counter:
        iven = 10000.0 / counter[elem]
        weight_dict[elem] = iven
        accum += iven
    for elem in weight_dict:
        weight_dict[elem] /= accum
    
    sample_weight = [ weight_dict[x] for x in label_list ]
    return sample_weight


def load_data(load_data_path):
        """
        从硬盘加载模型，字典等
        """
        data = None
        logger.info("load data in %s ..." % (load_data_path,))
        with open(load_data_path, 'rb') as fid:
            data = cPickle.load(fid)
        logger.info("load data %s finish" % (load_data_path,))
        return data

def save_data(data,save_data_path):
        """
        持久化模型，字典等
        """
        logger.info("save data in %s ..." % (save_data_path,))
        with open(save_data_path, 'wb') as fid:
            cPickle.dump(data, fid)
        logger.info("save data %s finish" % (save_data_path,))
        return True
    
def calculate_result(actual,pred):
    """
    计算precious
    """
    m_accuracy = metrics.accuracy_score(actual,pred);  
    print 'accuracy:{0:.3f}'.format(m_accuracy)  
    """
    m_precision = metrics.precision_score(actual,pred);  
    m_recall = metrics.recall_score(actual,pred);  
    print 'predict info:'  
    print 'precision:{0:.3f}'.format(m_precision)  
    print 'recall:{0:0.3f}'.format(m_recall);  
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred))
    """
    
def calculate_recall(actual,pred,specific_label):
    """
    计算某个类别的召回率
    """
    pass    

def get_confusion_matrix(actual,pred):
    format_matrix = []
    labels = list(set(actual))
    conf_mat = metrics.confusion_matrix(actual,pred,labels = labels)
    
    format_matrix.append("confusion_matrix(left labels: y_true, up labels: y_pred):")
    format_matrix.append("labels\t" + "\t".join(labels))
    for i in range(len(conf_mat)):
        row_list = [labels[i],]
        for j in range(len(conf_mat[i])):
            row_list.append(str(conf_mat[i][j]))
        format_matrix.append("\t".join(row_list))
    return format_matrix    

def get_classification_report(actual,pred):
    return metrics.classification_report(actual,pred)


def output_diff_result(raw_feature_list,actual_list,pred_list,output_file_path):
    assert len(raw_feature_list) == len(actual_list) == len(pred_list)
    with open(output_file_path,"w") as output_f:
        output_f.write("[is_true],[actual],[pred],[feature]\n")
        for index,feature in enumerate(raw_feature_list):
            print(type(feature))
            print(type(feature.encode("utf-8")))
            output_f.write("%s,%s,%s," % ((actual_list[index]==pred_list[index]),actual_list[index],pred_list[index]))
            output_f.write(feature.encode("utf-8"))
            output_f.write("\n")
            #output_f.write("%s,%s,%s\n" % ((actual_list[index]==pred_list[index]),actual_list[index],pred_list[index]))
        
