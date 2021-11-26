# -*- coding: utf-8 -*-
'''
输出感知损失结果
'''
# @Time    : 2021/11/22 16:38
# @Author  : LINYANZHEN
# @File    : model_test.py
import os.path
import shutil

import torch
import torch.backends.cudnn as cudnn
import utils
from SubPixelConvolution import model
from torchvision import models
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    features = models.vgg16(pretrained=True).features
    PerceptualModel = nn.Sequential()
    for i in range(9):
        # if i in (1, 3, 6, 8):
        #     continue
        # if i in (4, 9, 16, 23, 30):
        #     continue
        PerceptualModel.add_module(str(i), features[i])
    PerceptualModel.cuda()
    # 读图
    checkpoint_path = "checkpoint/content/SuperResolution/checkpoint/pl/SimpleSR_Upsample_with_PL_relu2_2_x4"

    origin_img_path = os.path.join(checkpoint_path, "test_99_x4.png")
    test_img_path = os.path.join(checkpoint_path, "test.png")
    origin_save_path = os.path.join(checkpoint_path, "test")
    test_save_path = os.path.join(checkpoint_path, "test_99_x4")
    if os.path.exists(origin_save_path):
        shutil.rmtree(origin_save_path)
    os.mkdir(origin_save_path)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.mkdir(test_save_path)

    origin_img = Variable(ToTensor()(Image.open(origin_img_path))).unsqueeze(0).cuda()
    test_img = Variable(ToTensor()(Image.open(test_img_path))).unsqueeze(0).cuda()
    output = PerceptualModel(origin_img).squeeze(0)
    print(output)
    for i in range(output.size()[0]):
        out = utils.tensor_to_image(output[i])
        out.save(os.path.join(origin_save_path, "{}.png".format(i)))
    output = PerceptualModel(test_img).squeeze(0)
    for i in range(output.size()[0]):
        out = utils.tensor_to_image(output[i])
        out.save(os.path.join(test_save_path, "{}.png".format(i)))
    # output = utils.tensor_to_image(output)
    # output.save("img_test/test_perceptual.png")
