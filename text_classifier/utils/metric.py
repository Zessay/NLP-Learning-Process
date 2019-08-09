#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:14
# @Author  : Zessay



class Metric(object):
    def __init__(self, pred_y, true_y, labels=None):
        self.pred_y = pred_y.ravel()
        self.true_y = true_y.ravel()
        self.labels = labels

    @classmethod
    def mean(cls, item: list) -> float:
        '''
        定义计算列表元素均值的函数
        '''
        res = sum(item) / len(item) if len(item) > 0 else 0
        return round(res, 5)

    def accuracy(self):
        '''
        计算二类和多类的准确率
        '''
        p = self.pred_y
        t = self.true_y
        if isinstance(p[0], list):
            p = [item[0] for item in p]
        corr = 0
        for i in range(len(p)):
            if p[i] == t[i]:
                corr += 1
        acc = corr / len(p) if len(p) > 0 else 0
        return round(acc, 5)

    def binary_precision(self, positive=1):
        '''
        二类精确率的计算
        '''
        p = self.pred_y
        t = self.true_y
        if isinstance(p[0], list):
            p = [item[0] for item in p]
        corr = 0
        pred_corr = 0
        for i in range(len(p)):
            if p[i] == positive:
                pred_corr += 1
                if p[i] == t[i]:
                    corr += 1
        prec = corr / pred_corr if pred_corr > 0 else 0
        return round(prec, 5)

    def binary_recall(self, positive=1):
        '''
        二类召回率的计算
        '''
        p = self.pred_y
        t = self.true_y
        if isinstance(p[0], list):
            p = [item[0] for item in p]
        corr = 0
        true_corr = 0
        for i in range(len(p)):
            if t[i] == positive:
                true_corr += 1
                if p[i] == t[i]:
                    corr += 1
        rec = corr / true_corr if true_corr > 0 else 0
        return round(rec, 5)

    def binary_f_beta(self, beta=1.0, positive=1):
        '''
        二类的f_beta的计算
        '''
        precision = self.binary_precision(positive)
        recall = self.binary_recall(positive)
        try:
            f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        except:
            f_b = 0
        return round(f_b, 5)

    def multi_precision(self):
        '''
        多类精确率的计算
        '''
        precisions = [self.binary_precision(label) for label in self.labels]
        prec = Metric.mean(precisions)
        return round(prec, 5)

    def multi_recall(self):
        '''
        多类召回率的计算
        '''
        recalls = [self.binary_recall(label) for label in self.labels]
        rec = Metric.mean(recalls)
        return round(rec, 5)

    def multi_f_beta(self, beta=1.0):
        '''
        多类f_beta的计算
        '''
        f_betas = [self.binary_f_beta(beta, label) for label in self.labels]
        f_beta = Metric.mean(f_betas)
        return round(f_beta, 5)

    def get_binary_metrics(self, f_beta=1.0):
        '''
        得到二类的性能指标
        '''
        metrics = {"accuracy": self.accuracy(), "recall": self.binary_recall(),
                   "precision": self.binary_precision(), "f_beta": self.binary_f_beta(f_beta)}
        return metrics

    def get_multi_metrics(self, f_beta=1.0):
        '''
        得到多类的性能指标
        '''
        metrics = {"accuracy": self.accuracy(), "recall": self.multi_recall(),
                   "precision": self.multi_precision(), "f_beta": self.multi_f_beta(f_beta)}
        return metrics
