#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: data.py
@time: 2019/10/10 20:44
@description: 构造词表，将单词和标记逐句存入列表中
'''

import os
import codecs


def build_corpus(split, make_vocab=True, data_dir="/root/zhushuai/data/nerdata"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            line = line.strip()
            if line:
                word, tag = line.split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {'<pad>': 0, '<unk>': 1}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

# def build_corpus(split, make_vocab=True, data_dir="/root/zhushuai/data/nerdata"):
#     '''读取数据并构建词表'''
#     assert split in ['train', 'dev', 'test']

#     word_lists = []
#     tag_lists = []
#     with codecs.open(os.path.join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
#         word_list = []
#         tag_list = []
#         for line in f:
#             line = line.strip()
#             if line:
#                 word, tag = line.split()
#                 word_list.append(word)
#                 tag_list.append(tag)
#             else:
#                 word_lists.append(word_list)
#                 tag_lists.append(tag_list)
#                 word_list = []
#                 tag_list = []

#     # 如果make_vocab为True，还需要返回word2id和tag2id
#     if make_vocab:
#         word2id = build_map(word_lists)
#         tag2id = build_map(tag_lists)
#         return word_lists, tag_lists, word2id, tag2id
#     else:
#         return word_lists, tag_lists


# def build_map(lists):
#     '''
#     构造单词到索引的映射
#     :param lists:
#     :return:
#     '''
#     maps = {'<pad>': 0, '<unk>': 1}
# #     maps = {}
#     for list_ in lists:
#         for e in list_:
#             if e not in maps:
#                 maps[e] = len(maps)
#     return maps