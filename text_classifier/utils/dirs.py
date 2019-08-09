#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 下午2:18
# @Author  : Zessay

import os 


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as e:
        print("Creating directories error: {}".format(e))
        exit(-1)