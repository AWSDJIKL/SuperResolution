# -*- coding: utf-8 -*-
'''
扩散模型，用于图片去噪
参考：https://arxiv.org/pdf/2009.00713.pdf
'''
# @Time    : 2021/11/3 0:12
# @Author  : LINYANZHEN
# @File    : model.py
import torch
import torch.nn as nn
import numpy as np


def Feature_wise_Affine(a, b1, b2):
    return a * b1 + b2


class UBlock(nn.Module):
    def __init__(self, input_channel, output_channel, scale_factor):
        super(UBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.conv3 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.conv4 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.conv1x1 = nn.Conv2d(input_channel, output_channel, (1, 1))
        self.lrelu = nn.LeakyReLU()
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.Feature_wise_Affine = Feature_wise_Affine

    def forward(self, x, b1, b2):
        identity = x

        x = self.lrelu(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.Feature_wise_Affine(x, b1, b2)
        x = self.lrelu(x)
        x = self.conv2(x)

        identity = self.upsample(identity)
        identity = self.conv1x1(identity)

        x += identity
        identity = x

        x = self.Feature_wise_Affine(x, b1, b2)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.Feature_wise_Affine(x, b1, b2)
        x = self.lrelu(x)
        x = self.conv4(x)

        x += identity
        return x


class DBlock(nn.Module):
    def __init__(self, input_channel, output_channel, scale_factor):
        '''
        下采样模块

        :param input_channel: 输入维度
        :param output_channel: 目标输出维度
        '''
        super(DBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, 3))
        self.conv2 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.conv3 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.conv1x1 = nn.Conv2d(input_channel, output_channel, (1, 1))
        self.lrelu = nn.LeakyReLU()
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        identity = x

        x = self.downsample(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)

        identity = self.conv1x1(identity)
        identity = self.downsample(identity)

        x += identity
        return x


class FiLM(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(FiLM, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, 3))
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, (3, 3))
        self.conv3 = nn.Conv2d(output_channel, output_channel, (3, 3))

    def forward(self, x, alpha):
        x = self.conv(x)
        x = self.lrelu(x)
        # alpha经过位置编码器
        # x += alpha
        # 将x拆分，在通道维度上拆分成两个大小相同的
        b1, b2 = x.split(x.size()(1) / 2, dim=0)
        b1 = self.conv2(b1)
        b2 = self.conv3(b2)
        return b1, b2


class DiffusionModel(nn.Module):
    '''
    噪音生成模型
    '''

    N = 5
    beta = np.logspace(0, 1, N)

    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.conv_x1 = nn.Conv2d(3, 768, (3, 3))
        self.ublock1 = UBlock(768, 512, 5)
        self.ublock2 = UBlock(512, 512, 5)
        self.ublock3 = UBlock(512, 256, 3)
        self.ublock4 = UBlock(256, 128, 2)
        self.ublock5 = UBlock(128, 128, 2)
        self.conv_x2 = nn.Conv2d(128, 3, (3, 3))

        self.conv_y = nn.Conv2d(3, 32, (5, 5))
        self.dblck1 = DBlock(32, 128, 2)
        self.dblck2 = DBlock(128, 128, 2)
        self.dblck3 = DBlock(128, 256, 3)
        self.dblck4 = DBlock(256, 512, 5)

        self.film1 = FiLM(32, 128)
        self.film2 = FiLM(128, 128)
        self.film3 = FiLM(128, 256)
        self.film4 = FiLM(256, 512)
        self.film5 = FiLM(512, 512)

    def forward(self, x, y, n):
        '''

        :param x: 没有噪声的图片
        :param y: 第n次迭代的带噪声的图片
        :param n: 轮次数，影响alpha的值
        :return: 模拟生产的噪声
        '''

        alpha = 1
        for i in range(n):
            alpha *= 1 - self.beta[i]

        y = self.conv_y(y)
        film1_b1, film1_b2 = self.film1(y, alpha)
        y = self.dblck1(y)
        film2_b1, film2_b2 = self.film2(y, alpha)
        y = self.dblck2(y)
        film3_b1, film3_b2 = self.film3(y, alpha)
        y = self.dblck3(y)
        film4_b1, film4_b2 = self.film4(y, alpha)
        y = self.dblck4(y)
        film5_b1, film5_b2 = self.film5(y, alpha)

        x = self.conv_x1(x)
        x = self.ublock1(x, film1_b1, film1_b2)
        x = self.ublock2(x, film2_b1, film2_b2)
        x = self.ublock3(x, film3_b1, film3_b2)
        x = self.ublock4(x, film4_b1, film4_b2)
        x = self.ublock5(x, film5_b1, film5_b2)
        noise = self.conv_x2(x)

        return noise


class Noise_loss(nn.Module):
    '''
    用于训练噪声生成模型的损失函数
    '''

    def __init__(self):
        super(Noise_loss, self).__init__()

    def forward(self, x):
        # 首先生成与输入大小相同的高斯噪声
        x = 1


class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()
        self.noise_generator = DiffusionModel()

    def forward(self, x, y):
        pass
