#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:13
# @Author  : Zessay

import tensorflow as tf
import os

class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_sumary_writer = tf.summary.FileWriter(os.path.join(self.config['summary_dir'], "train"),
                                                         self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config['summary_dir'], "test"))

    # 保存scalars和images
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        '''
        step: 表示summary的时间步
        summarizer: 表示使用 train 还是 test
        scope: 表示变量空间名
        summaries_dict: 表示要summaries的值，格式是(tag, value)
        '''
        summary_writer = self.train_sumary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder(tf.float32, shape=value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder("float32",
                                                                            [None] + list(value.shape[1:]),
                                                                            name=tag)
                        if len(value.shape) <= 1:
                            ## 添加标量
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            ## 添加为图片
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag],
                                                      {self.summary_placeholders[tag]: value}))
                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()