# -*- coding: utf-8 -*-
'''
模块代码测试
'''
# @Time    : 2021/8/5 15:43
# @Author  : LINYANZHEN
# @File    : test_script.py
import os
import datetime
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import h5py


class test_net(nn.Module):
    def __init__(self):
        super(test_net, self).__init__()
        self.conv = nn.Conv2d(3, 3, (3, 3), (1, 1), (5, 5), (5, 5))

    def forward(self, x):
        print(x.size())
        x = self.conv(x)
        print(x.size())
        return x


if __name__ == '__main__':
    model = test_net()
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
