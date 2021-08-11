# -*- coding: utf-8 -*-
'''
模块代码测试
'''
# @Time    : 2021/8/5 15:43
# @Author  : LINYANZHEN
# @File    : model_test.py
import os

import torch
import numpy as np
from PIL import Image
import h5py

if __name__ == '__main__':
    path="dataset/BSD500/BSR/BSDS500"
    for root, dirs, files in os.walk("dataset/BSD500/BSR/BSD500/data/images"):
        print(files)