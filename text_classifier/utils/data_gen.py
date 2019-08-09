#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:11
# @Author  : Zessay

import numpy as np

class DataGenerator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(y)

    def next_batch(self, batch_size):
        '''
        生成每一个batch的数据集
        '''
        idx = np.random.choice(self.length, batch_size)
        yield self.x[idx], self.y[idx]

    def iter_all(self, batch_size):
        '''
        按照batch迭代所有数据
        '''
        numBatches = self.length // batch_size
        for i in range(numBatches):
            start = i * batch_size
            end = start + batch_size
            batchX = np.array(self.x[start:end], dtype='int64')
            batchY = np.array(self.y[start:end], dtype="float32")
            yield batchX, batchY
