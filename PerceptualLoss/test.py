# -*- coding: utf-8 -*-
'''
单元测试
'''
# @Time    : 2021/8/5 15:43
# @Author  : LINYANZHEN
# @File    : model_test.py

import torch
import numpy as np
from PIL import Image
import h5py

if __name__ == '__main__':
    model = torch.load("../checkpoint/sub_pixel_convolution.pth")
    print(model.__class__.__name__)
