#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:14
# @Author  : Zessay

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np


class Metric:
    def __init__(self, y_proba, y_true, config):
        self.y_proba = y_proba
        self.y_true = y_true.ravel().astype("int")
        self.config = config

        ## 判断标签
        if len(np.unique(self.y_true)) == 2:
            self.y_pred = (self.y_proba.ravel() > config.threshold).astype("int")
        else:
            self.y_pred = np.argmax(self.y_proba, axis=1).ravel()

    def accuracy(self):
        '''
        计算准确率
        '''
        acc = accuracy_score(self.y_true, self.y_pred)
        return round(acc, 5)

    def precision(self):
        '''
        计算精确度
        '''
        if len(np.unique(self.y_true)) == 2:
            prec = precision_score(self.y_true, self.y_pred)
        else:
            prec = precision_score(self.y_true, self.y_pred, average="micro")
        return round(prec, 5)

    def recall(self):
        '''
        计算召回率
        '''
        if len(np.unique(self.y_true)) == 2:
            rec = recall_score(self.y_true, self.y_pred)
        else:
            rec = recall_score(self.y_true, self.y_pred, average="micro")
        return round(rec, 5)

    def f_score(self):
        if len(np.unique(self.y_true)) == 2:
            f = f1_score(self.y_true, self.y_pred)
        else:
            f = f1_score(self.y_true, self.y_pred, average="micro")
        return round(f, 5)

    def auc(self):
        uni = len(np.unique(self.y_true))
        if uni <= 2:
            a = roc_auc_score(self.y_true, self.y_proba.ravel())
        else:
            y_true = np.eye(np.max(self.y_true) + 1)[self.y_true]
            a = roc_auc_score(y_true, self.y_proba)
        return round(a, 5)

    def _gini(self, y_true, y_proba):
        assert (len(y_true) == len(y_proba))
        cat = np.asarray(np.c_[y_true, y_proba, np.arange(len(y_true))], dtype=np.float)
        cat = cat[np.lexsort((cat[:, 2], -1 * cat[:, 1]))]  # 按照概率从大到小的顺序，如果相同则再按照索引
        totalLoss = cat[:, 0].sum()
        giniSum = cat[:, 0].cumsum().sum() / totalLoss
        giniSum -= (len(y_true) + 1) / 2
        return giniSum / len(y_true)

    def gini_norm(self):
        return self._gini(self.y_true, self.y_proba.ravel()) / self._gini(self.y_true, self.y_true)

    def get_metrics(self):
        if len(np.unique(self.y_true)) == 2:
            metrics = {"accuracy": self.accuracy(), "precision": self.precision(),
                       "recall": self.recall(), "f_score": self.f_score(),
                       "auc": self.auc(), "gini_norm": self.gini_norm()}
        else:
            metrics = {"accuracy": self.accuracy(), "precision": self.precision(),
                       "recall": self.recall(), "f_score": self.f_score(),
                       "auc": self.auc()}

        return metrics