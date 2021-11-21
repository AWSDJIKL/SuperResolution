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
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from PerceptualLoss import lossfunction
import torch.nn.functional as F

if __name__ == '__main__':
    # image_path_list = ["img_test/test.png",
    #                    "img_test/test_JohnsonSR_without_PL_x4.png",
    #                    "img_test/test_JohnsonSR_with_PL_x4.png",
    #                    "img_test/test_bicubic_x4.png"]
    image_path_list = ["img_test/blow_up_deatial/target_bird.jpg",
                       "img_test/blow_up_deatial/target_bird_bicubic_x4.jpg",
                       "img_test/blow_up_deatial/target_bird_johnson_with_pl_x4.jpg",
                       "img_test/blow_up_deatial/github_sample.jpg"]
    image_title_list = ["HR",
                        "bicubic",
                        "ours",
                        "github_sample"]
    image_list = []
    for i in range(len(image_path_list)):
        input_image = Image.open(image_path_list[i])
        image_list.append(utils.blow_up_details(input_image, (125, 200), (50, 50), 2))
    utils.image_concat(image_list, image_title_list, "img_test/blow_up_deatial/bird_JohnsonSR_x4.jpg")
