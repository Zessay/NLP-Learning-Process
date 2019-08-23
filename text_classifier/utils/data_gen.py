#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:11
# @Author  : Zessay

import numpy as np
from collections import Counter

class DataGenerator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(y)
        ## 计算不同类别的比例
        unique = Counter(self.y.ravel())
        self.ratio = [(key, value / self.length) for key, value in unique.items()]
        self.indices = []
        for key, _ in self.ratio:
            index = np.where(y.ravel() == key)
            self.indices.append(index)

    def next_batch(self, batch_size):
        '''
        生成每一个batch的数据集
        '''
        choose = np.array([])
        for i in range(len(self.indices)):
            idx = np.random.choice(self.indices[i][0],
                                   max(1, min(len(self.indices[i][0]), int(batch_size * self.ratio[i][1]))))
            choose = np.append(choose, idx)
        choose = np.random.permutation(choose).astype("int64")
        yield self.x[choose], self.y[choose]

    def iter_all(self, batch_size):
        '''
        按照batch迭代所有数据
        '''
        numBatches = self.length // batch_size + 1
        for i in range(numBatches):
            start = i * batch_size
            end = min(start + batch_size, self.length)
            batchX = np.array(self.x[start:end], dtype='int64')
            batchY = np.array(self.y[start:end], dtype="float32")
            yield batchX, batchY
