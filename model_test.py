# -*- coding: utf-8 -*-
'''
测试模型效果
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : model_test.py
import os

import torch
import torch.backends.cudnn as cudnn
import utils
from PerceptualLoss import model

if __name__ == '__main__':
    upscale_factor = 3
    test_image_path = "img_test/test.jpg"
    state_dict_path = "checkpoint/PerceptualLoss.pth"
    cudnn.benchmark = True
    model = model.Sub_pixel_conv(upscale_factor)
    for name, parameters in torch.load(state_dict_path).items():
        if name in model.state_dict().keys():
            model.state_dict()[name].copy_(parameters)
        else:
            raise KeyError(name)
    utils.test_model(model, test_image_path, upscale_factor)
