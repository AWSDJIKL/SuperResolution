# -*- coding: utf-8 -*-
'''
测试模型效果
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : test.py
import os

import torch
import torch.backends.cudnn as cudnn
import utils
from SubPixelConvolution import model

# from ResizeConvolution import model

if __name__ == '__main__':
    upscale_factor = 3
    # test_image_path = "img_test/test.png"
    # test_image_path = "img_test/baby/baby.png"
    test_image_path = "img_test/baboon/baboon.png"

    cudnn.benchmark = True
    state_dict_path = "checkpoint/Residual_SPC_with_mix_PL_relu2_2_best.pth"
    # state_dict_path = "checkpoint/Residual_SPC_without_PL_best.pth"
    save_name = "Residual_SPC_with_mix_PL"
    model = model.Residual_SPC(upscale_factor)
    for name, parameters in torch.load(state_dict_path).items():
        if name in model.state_dict().keys():
            model.state_dict()[name].copy_(parameters)
        else:
            raise KeyError(name)
    # utils.test_model(model, "without_PL_final", test_image_path, upscale_factor)
    utils.test_model(model, test_image_path, upscale_factor, save_name)
