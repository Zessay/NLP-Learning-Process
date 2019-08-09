#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:12
# @Author  : Zessay

import tensorflow as tf

class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.data = data
        self.sess = sess
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train_all(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config['num_epochs'], 1):
            print(f"\n当前正处于第{cur_epoch + 1}次迭代")
            self.train_epoch()
            ## 将对应的epoch+1
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        '''
        实现一个epoch训练的代码
        - 在config规定的迭代次数上迭代，调用train_step
        - 添加summary
        '''
        raise NotImplementedError

    def train_step(self):
        '''
        实现单步训练的逻辑代码
        '''
        raise NotImplementedError
