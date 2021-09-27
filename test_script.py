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
import utils
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from PerceptualLoss import lossfunction
import torch.nn.functional as F

if __name__ == '__main__':
    k1 = torch.Tensor([[[[i for i in range(448)] for i in range(448)] for i in range(3)] for i in range(4)])
    k2 = torch.Tensor([[[5 for i in range(50)] for j in range(3)] for i in range(4)]).cuda()
    k2 = k2.unsqueeze(3)
    k2 = F.pad(k2, [448 // 2 - 1, 448 // 2, (448 - 50) // 2, (448 - 50) // 2], "constant", k2.mean())
    print(k2.size())
    kernel = utils.create_kernel(3, 3)
    print(kernel.size())
    k2 = F.conv2d(k2, weight=kernel, stride=1, padding=224)
    print(k2.size())
    # print(k1.size())
    # print(k2.size())
    # print((k1 + k2).size())
    # print((k1 * k2))
