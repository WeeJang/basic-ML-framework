#!/usr/bin/env python2
#-*- coding:utf-8 -*-

import os,sys
import json
import logging

import numpy as np
from sklearn import linear_model
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.externals import joblib

CUR_DICT_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(CUR_DICT_PATH,"../metrics"))
sys.path.append(os.path.join(CUR_DICT_PATH,"../word-cutter"))

from helper_functions import calculate_result,save_data,load_data
from Cutter import Cutter

logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel("DEBUG")

class FeatureMaker(object):
    """
        Basic class for feature-engine,
        Infrastructure provided:
            1)CountVectorizer-like
            2)TfidfTransformer-like
            3)PCA-like
        you could use these infrastructure to create feature,
        FeatureMaker will automatic persisted related data(like `vocabulary_` for CountVectorizer,`model` for PCA,eg).
        What only you should do is inherit this `FeatureMaker` class and implement the `transfrom` method(interface).

        Basic Usage:
            class MyFeatureMaker(FeatureMaker):
                def __init__(self,conf_path,service_type,is_train = True):
                    FeatureMaker.__init__(conf_path,service_type,is_train)
                    #your other init code
                #override
                def transform(self,raw_data_list):
                    #your feature-engine-code

    """
    def __init__(self,conf_path,service_type,is_train = True):
        """Construction
        -conf_path : configure dictionary path,str type
        -service_type : service type name ,str type
        -is_train : is training stage flag, boolean type,defualt is True
        """
        self.__conf_path__ = conf_path
        self.__service_type__ = service_type
        self.__is_train__ = is_train
        self.__cutter__ = Cutter(self.__conf_path__)
        self.__tfidf_model_dict__ = {}
        self.__countvec_model_dict__ = {}
        self.__pca_model_dict__ = {}

    def __del__(self):
        """Destruction,do necessary persisting job if in training stage
        """
        if self.__is_train__:
            logger.error("fuck!!!")
            self.__save_countvec_vocab__()
            self.__save_pca_model__()
            self.__save_idf_model__()
        
    def __get_countvec_vocab_path__(self,countvec_model_name):
        """CountVectorizer vocabulary file path
        """
        return "%s/%s.%s.count_vectorizer.vocabulary.pkl" % (self.__conf_path__,self.__service_type__,countvec_model_name)

    def __get_pca_model_path__(self,pca_model_name):
        """PCA model file path
        """
        return "%s/%s.%s.pca.pkl" % (self.__conf_path__,self.__service_type__,pca_model_name)
   
    def __get_tfidf_model_path__(self,tfidf_model_name):
        """tfidf model file path
        """
        return "%s/%s.%s.idf.pkl" % (self.__conf_path__,self.__service_type__,tfidf_model_name)
    
    def __save_countvec_vocab__(self):
        """save CountVectorizer vocabulary file
        """
        for model_name in self.__countvec_model_dict__:
            countvec_vocab_path = self.__get_countvec_vocab_path__(model_name)
            save_data(self.__countvec_model_dict__[model_name].vocabulary_,countvec_vocab_path)

    def __save_pca_model__(self):
        """save PCA model file
        """
        for model_name in self.__pca_model_dict__:
            pca_model_path = self.__get_pca_model_path__(model_name)
            save_data(self.__pca_model_dict__[model_name],pca_model_path)

    def __save_idf_model__(self):
        """save idf model file
        """
        for model_name in self.__tfidf_model_dict__:
            tfidf_model_path = self.__get_tfidf_model_path__(model_name)
            save_data(self.__tfidf_model_dict__[model_name],tfidf_model_path)

    def __countvec_fit_transform__(self,countvec_model_name,raw_documents):
        """CountVectorizer fit_transform
        -countvec_model_name:CountVectorizer model name, str type
	    -raw_documents:An iterable which yields either str, unicode or file objects.
        """
        logger.debug("raw_documents %s" % (raw_documents))
        decode_error = "ignore"
        analyzer = "word"
        ngram_range = (1,3)
        if self.__is_train__:
            if countvec_model_name in self.__countvec_model_dict__:
                raise Exception("CountVectorizer[%s] exist,re-training will override" % (countvec_model_name,))
            countvec_model = CountVectorizer(decode_error=decode_error,analyzer=analyzer,ngram_range=ngram_range)
            self.__countvec_model_dict__[countvec_model_name] = countvec_model
            return countvec_model.fit_transform(raw_documents)

        else:
            countvec_model = self.__countvec_model_dict__.get(countvec_model_name)
            if countvec_model is None:
                countvec_path = self.__get_countvec_vocab_path__(countvec_model_name)
                vocab = load_data(countvec_path)
                countvec_model = CountVectorizer(decode_error=decode_error,vocabulary=vocab,analyzer=analyzer,ngram_range=ngram_range)
                self.__countvec_model_dict__[countvec_model_name] = countvec_model
            return countvec_model.transform(raw_documents)

    def __pca_fit_transform__(self,pca_model_name,raw_tfidf_matrix,dim = None,n_iter = 1000):
        """PCA fit transform
        -pca_model_name:PCA model name,str type
        -raw_tfidf_matrix:tfidf matrix,matrix type
        -dim:PCA dim,int type
        -n_iter:PCA n_iter,int type
        NOTE:
            when FeatureMaker is in training stage,dim must be assigned.
        """
        if self.__is_train__:
            assert dim is not None,"when training,`dim` must be assigned"
        n_iter = n_iter
    
        if self.__is_train__:
            if pca_model_name in self.__pca_model_dict__:
                raise Exception("PCAmodel[%s] exist,re-training will override" % (pca_model_name,))
            pca_model = decomposition.TruncatedSVD(n_components = dim, n_iter=n_iter)
            self.__pca_model_dict__[pca_model_name] = pca_model
            return pca_model.fit_transform(raw_tfidf_matrix)
        else:
            pca_model = self.__pca_model_dict__.get(pca_model_name)
            if pca_model is None:
                pca_model_path = self.__get_pca_model_path__(pca_model_name)
                pca_model = load_data(pca_model_path)
                self.__pca_model_dict__[pca_model_name] = pca_model
            return pca_model.transform(raw_tfidf_matrix)

    def __tfidf_fit_transform__(self,tfidf_model_name,raw_wordcount_matrix):
        if self.__is_train__:
            if tfidf_model_name in self.__tfidf_model_dict__:
                raise Exception("tfidf_model[%s] exist,re-training will override" % (tfidf_model_name,))
            tfidf_model = TfidfTransformer()
            self.__tfidf_model_dict__[tfidf_model_name] = tfidf_model
            return tfidf_model.fit_transform(raw_wordcount_matrix)
        else:
            tfidf_model = self.__tfidf_model_dict__.get(tfidf_model_name)
            if tfidf_model is None:
                tfidf_model_path = self.__get_tfidf_model_path__(tfidf_model_name)
                tfidf_model = load_data(tfidf_model_path)
                self.__tfidf_model_dict__[tfidf_model_name] = tfidf_model
            return tfidf_model.transform(raw_wordcount_matrix)


    def LSA_vectorize(self,model_name,raw_document,dim = None):
        if self.__is_train__:
            assert dim is not None,"when training,`dim` must be assigned"
    
        word_count_matrix = self.__countvec_fit_transform__(model_name,raw_document)
        tfidf_matrix = self.__tfidf_fit_transform__(model_name,word_count_matrix)
        return tfidf_matrix.toarray()
        """
        lsa_matrix = None
        if self.__is_train__:
            lsa_matrix = self.__pca_fit_transform__(model_name,tfidf_matrix,dim = dim)
        else:
            lsa_matrix = self.__pca_fit_transform__(model_name,tfidf_matrix)
        return lsa_matrix
        """
    
    def transform(self,raw_data_list):
        """Interface of FeatureMaker
        -raw_data_list : list type, the type of elem is dict.
            for example: [{"raw_feature_1" : f_value_1 , "raw_feature_2" : f_value_2 },\
                          {"raw_feature_1" : f_value_1 , "raw_feature_2" : f_value_2 },\
                            ... ... \
                          {"raw_feature_1" : f_value_1 , "raw_feature_2" : f_value_2 },\
                     ]
        """
        pass

