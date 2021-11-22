# -*- coding: utf-8 -*-
'''
自定义损失函数
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : lossfunction.py
import copy

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from SubPixelConvolution import model
import collections
from torchvision import transforms

vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
output_layers = {
    "relu1_2": 2,
    "relu2_2": 5,
    "relu3_3": 9,
    "relu4_3": 13,
}


class PerceptualVGG16(nn.Module):
    def __init__(self, output_layer="relu2_2"):
        super(PerceptualVGG16, self).__init__()
        self.features = vgg_make_layers(vgg16[:output_layers[output_layer]])
        pretrain_state_dict = models.vgg16(True, True).state_dict()
        # print(pretrain_state_dict.keys())
        # print(self.state_dict().keys())
        # 加载预训练权重
        for name, parameters in self.state_dict().items():
            if name in pretrain_state_dict.keys():
                # print(name)
                self.state_dict()[name].copy_(pretrain_state_dict[name])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return x


def vgg_make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=(1, 1))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class vgg16_loss(nn.Module):
    def __init__(self, output_layer="relu2_2"):
        '''
        感知损失(vgg16)

        :param output_layer: 在第几层输出 可选项：["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        '''
        # 定义必要的超参数
        super(vgg16_loss, self).__init__()
        # self.PerceptualModel = PerceptualVGG16(output_layer).cuda()
        # self.PerceptualModel.eval()
        #
        # # self.PerceptualModel = Vgg16().cuda()
        # # self.PerceptualModel.eval()
        # for param in self.PerceptualModel.parameters():
        #     param.requires_grad = False
        features = models.vgg16(pretrained=True).features
        print(features)
        self.PerceptualModel = nn.Sequential()
        for i in range(9):
            if i in (4, 9, 16, 23, 30):
                continue
            self.PerceptualModel.add_module(str(i), features[i])
        self.PerceptualModel.cuda()

    def forward(self, pred, y):
        # pred = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(pred)
        # y = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(y)
        # 定义计算过程
        # 输出结果与真实结果一起经过感知损失模型
        Perceptual_pred = self.PerceptualModel(pred)
        Perceptual_y = self.PerceptualModel(y)

        (batch_size, c, h, w) = y.size()
        # perceptual_loss = (torch.sum((Perceptual_pred - Perceptual_y) ** 2) ** 0.5) / (batch_size * c * h * w)
        perceptual_loss = torch.nn.MSELoss()(Perceptual_pred, Perceptual_y) / (batch_size * c * h * w)
        # perceptual_loss = 0
        # for i in range(4):
        #     (batch_size, c, h, w) = Perceptual_pred[i].size()
        #     perceptual_loss += torch.nn.MSELoss()(Perceptual_pred[i], Perceptual_y[i]) / (batch_size * c * h * w)
        # mse_loss = (torch.sum((pred - y) ** 2) ** 0.5)
        # mse_loss = torch.nn.MSELoss()(pred, y)
        # loss = mse_loss + perceptual_loss
        loss = perceptual_loss
        pred.cpu()
        y.cpu()
        torch.cuda.empty_cache()

        return loss


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out
