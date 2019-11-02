#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: utils
@time: 2019/10/10 21:04
@description: 定义一些工具函数
'''
import pickle
import codecs

def merge_maps(dict1, dict2):
    '''
    用于合并两个word2id或者两个tag2id
    :param dict1:
    :param dict2:
    :return:
    '''
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1

def save_model(model, file_name):
    '''
    用于保存模型
    :param model:
    :param file_name:
    :return:
    '''
    with codecs.open(file_name, "wb") as f:
        pickle.dump(model, f)

def load_model(file_name):
    '''
    用于加载模型
    :param file_name:
    :return:
    '''
    with codecs.open(file_name, "rb") as f:
        model = pickle.load(f)
    return model

def extend_maps(word2id, tag2id):
    '''
    如果是加了CRF的LSTM模型需要加入<start>和<end>
    :param word2id:
    :param tag2id:
    :return:
    '''
    word2id['<start>'] = len(word2id)
    word2id['<end>'] = len(word2id)
    tag2id['<start>'] = len(tag2id)
    tag2id['<end>'] = len(tag2id)

    return word2id, tag2id

def preprocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    '''
    对word和tag添加结束标志
    :param word_lists:
    :param tag_lists:
    :param test:
    :return:
    '''
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        ## 如果是测试集，就不需要加end token了
        if not test:
            tag_lists[i].append("<end>")
    return word_lists, tag_lists

def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list

