# -*- coding: utf-8 -*-
'''
缩放卷积，先用双线性插值将原图放大，然后用卷积层学习细节
'''
# @Time    : 2021/8/16 17:11
# @Author  : LINYANZHEN
# @File    : model.py

import torch.nn as nn
import typing

import torch.nn.functional as F


class ResidualRegularBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualRegularBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, (3, 3), (1, 1), (1, 1), dilation=(1, 1))
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, (3, 3), (1, 1), (2, 2), dilation=(2, 2))
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv3 = nn.Conv2d(output_channels, output_channels, (3, 3), (1, 1), (5, 5), dilation=(5, 5))
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.conv1x1 = nn.Conv2d(input_channels, output_channels, (1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        output += self.conv1x1(identity)
        output = self.relu(output)
        return output


class ResizeConvolution(nn.Module):
    def __init__(self, upscale_factor, in_channels=3):
        '''
        缩放卷积

        :param upscale_factor: 放大倍数
        :param in_channels: 输入图片其实通道数
        '''
        super(ResizeConvolution, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv1 = nn.Conv2d(in_channels, 64, (1, 1))
        self.block1 = ResidualRegularBlock(input_channels=64, output_channels=64)
        self.block2 = ResidualRegularBlock(input_channels=64, output_channels=64)
        self.block3 = ResidualRegularBlock(input_channels=64, output_channels=64)
        self.block4 = ResidualRegularBlock(input_channels=64, output_channels=64)
        # self.conv2 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 3, (1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode="bilinear", align_corners=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv2(x)
        return x
