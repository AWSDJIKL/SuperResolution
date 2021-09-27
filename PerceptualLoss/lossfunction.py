# -*- coding: utf-8 -*-
'''
自定义损失函数
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : lossfunction.py
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from SubPixelConvolution import model

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
                self.state_dict()[name].copy_(pretrain_state_dict[name])

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
        self.PerceptualModel = PerceptualVGG16(output_layer).cuda()
        self.PerceptualModel.eval()

    def forward(self, pred, y):
        # 定义计算过程
        # 输出结果与真实结果一起经过感知损失模型
        Perceptual_pred = self.PerceptualModel(pred)
        Perceptual_y = self.PerceptualModel(y)
        # 获取作为label的来自于预训练的深度神经网络中间层的输出的大小
        # print(pred.size())
        # print(y.size())
        (batch_size, c, h, w) = y.size()
        # perceptual_loss = (torch.sum((pred[0] - y[0]) ** 2) ** 0.5) / (c * h * w)
        # # 分开计算batch内每个样本的损失
        # for i in range(1, batch_size):
        #     perceptual_loss += (torch.sum((pred[i] - y[i]) ** 2) ** 0.5) / (c * h * w)
        perceptual_loss = (torch.sum((Perceptual_pred - Perceptual_y) ** 2) ** 0.5) / (batch_size * c * h * w)
        mse_loss = (torch.sum((pred - y) ** 2) ** 0.5)
        # mse_loss = torch.nn.MSELoss()(pred, y)
        loss = mse_loss + perceptual_loss
        # loss = perceptual_loss
        pred.cpu()
        y.cpu()
        torch.cuda.empty_cache()
        return loss


class Residual_SPC_loss(nn.Module):
    def __init__(self, model_path, upscale_factor=3):
        super(Residual_SPC_loss, self).__init__()
        self.loss_model = model.Residual_SPC(upscale_factor).cuda().eval()
        for name, parameters in torch.load(model_path).items():
            if name in self.loss_model.state_dict().keys():
                self.loss_model.state_dict()[name].copy_(parameters)
            else:
                raise KeyError(name)

    def forward(self, pred, y):
        pred = self.loss_model(pred)
        y = self.loss_model(y)
        (batch_size, c, h, w) = y.size()
        # # 分开计算batch内每个样本的损失
        loss = (torch.sum((pred - y) ** 2) ** 0.5) / (batch_size * c * h * w)
        pred.cpu()
        y.cpu()
        torch.cuda.empty_cache()
        return loss
