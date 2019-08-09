#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:06
# @Author  : Zessay

import tensorflow as tf

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config['checkpoint_dir'] + "/my_model", self.global_step_tensor)
        print("Model saved")

    def load(self, sess):
        ## 获取最近的chekpoint
        latest_checkpoint = tf.train.latest_checkpoint(self.config['checkpoint_dir'])
        if latest_checkpoint:
            print("Loading model checkpoint {} ... \n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # 表示每执行一个epoch，对应的变量+1
    def init_cur_epoch(self):
        with tf.variable_scope("cur_epoch"):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name="cur_epoch")
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        # 表示当前模型一共迭代的step
        ## 每次执行都需要放到trainer里面
        with tf.variable_scope("global_step"):
            self.global_step_tensor = tf.Variable(0, trainable=False, name="global_step")

    def init_saver(self):
        # 通常只需要在子类中拷贝下面的语句即可
        # self.saver = tf.train.Saver(max_to_keep=self.config['max_to_keep'])
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError