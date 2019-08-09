#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:08
# @Author  : Zessay

import pandas as pd
import numpy as np
import gensim
import os
from collections import Counter
import json

# 定义数据预处理类
class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config['dataSource']
        self._stopWordSource = config['stopWordSource']

        self._sequenceLength = config['sequenceLength']  # 设置序列的输入藏毒
        self._embeddingSize = config['embeddingSize']
        self._batchSize = config['batch_size']
        self._trainRate = config['train_size']

        self._stopWordDict = {}
        self.trainReviews = []
        self.trainLabels = []
        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None
        self.labelList = []

    def _readData(self, filePath):
        '''
        从csv文件中读取数据集
        '''
        df = pd.read_csv(filePath)
        if self.config['numClasses'] == 1:
            if "sentiment" in df.columns:
                labels = df["sentiment"].tolist()
            if "emotion" in df.columns:
                labels = df["emotion"].tolist()

        elif self.config['numClasses'] > 1:
            labels = df["rate"].tolist()

        review = df['review'].tolist()
        reviews = [line.strip().split() for line in review]

        return reviews, labels

    def _laeblToIndex(self, labels, label2idx):
        '''
        将标签转换为索引表示
        '''
        labelIds = [label2idx[label] for label in labels]
        return labelIds

    def _wordToIndex(self, reviews, word2idx):
        '''
        将词转换为索引表示
        '''
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds

    def _genTrainEvalData(self, x, y, word2idx, rate):
        '''
        生成训练集和验证集
        '''
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))

        trainIndex = int(len(x) * rate)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(y[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews, labels, path, prefix=""):
        '''
        生成向量和词汇-索引映射字典
        '''

        save_path = "../data/wordJson"
        target_word_dir = os.path.join(save_path, prefix + "_word2idx.json")
        target_label_dir = os.path.join(save_path, prefix + "_label2idx.json")

        try:
            word2idx = json.loads(target_word_dir)
            label2idx = json.loads(target_label_dir)
        
        except:
            allWords = [word for review in reviews for word in review]
            # 去掉停用词
            subWords = [word for word in allWords if word not in self.stopWordDict]
            wordCount = Counter(subWords)  # 统计各个词的词频
            sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

            # 去除低频词
            words = [item[0] for item in sortWordCount if item[1] >= 5]

            vocab, wordEmbedding = self._getWordEmbedding(words, path)
            self.wordEmbedding = wordEmbedding

            # print(len(vocab), vocab[10])
            word2idx = dict(zip(vocab, range(len(vocab))))

            uniqueLabel = list(set(labels))
            label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
            self.labelList = list(range(len(uniqueLabel)))

            # 将词汇表-索引映射表保存为json数据，之后inference时直接加载处理数据
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(target_word_dir, "w", encoding="utf8") as f:
                json.dump(word2idx, f)

            with open(target_label_dir, "w", encoding="utf8") as f:
                json.dump(label2idx, f)

        return word2idx, label2idx

    def _getWordEmbedding(self, words, path):
        '''
        按照数据集中的单词去除训练好的词向量
        '''
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(path, "wordvector.bin"),
                                                                  binary=True)

        vocab = []
        wordEmbedding = []

        # 添加"pad"和"UNK"
        vocab.append("PAD")
        vocab.append("UNK")

        wordEmbedding.append(np.zeros(self._embeddingSize))  # 表示对"PAD"用全0向量表示
        wordEmbedding.append(np.random.randn(self._embeddingSize))  # 对"UNK"用随机向量表示

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                pass

        return vocab, np.array(wordEmbedding)

    def _readStopword(self, stopWordPath):
        '''
        读取停用词
        '''
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 转换成字典的形式，使用hash查找效率更高
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self, path, prefix=""):
        '''
        初始化训练集和验证集
        prefix: 表示生成单词到索引的文件的前缀
        path: 表示wordvector文件的位置
        '''
        # 初始化停用词
        self._readStopword(self._stopWordSource)
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)
        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels, path, prefix)
        # 将标签和句子数值化
        labelIds = self._laeblToIndex(labels, label2idx)
        reviewsIds = self._wordToIndex(reviews, word2idx)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewsIds,
                                                                                    labelIds,
                                                                                    word2idx,
                                                                                    self._trainRate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels