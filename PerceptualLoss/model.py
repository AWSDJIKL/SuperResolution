# -*- coding: utf-8 -*-
'''
自定义模型
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : model.py

import torch.nn as nn
import typing


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, (3, 3), (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, (3, 3), (1, 1), (1, 1))
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv1x1 = nn.Conv2d(input_channels, output_channels, (1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        output += self.conv1x1(x)
        output = self.relu(output)
        return output


class Sub_pixel_conv(nn.Module):
    def __init__(self, upscale_factor, in_channels=3):
        '''
        亚像素卷积网络

        :param upscale_factor: 放大倍数
        '''
        super(Sub_pixel_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, (5, 5), (1, 1), (2, 2))
        self.block1 = ResidualBlock(input_channels=64, output_channels=64)
        self.block2 = ResidualBlock(input_channels=64, output_channels=64)
        self.block3 = ResidualBlock(input_channels=64, output_channels=64)
        self.block4 = ResidualBlock(input_channels=64, output_channels=64)
        self.conv2 = nn.Conv2d(64, in_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))

        self.ConvTranspose = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(3, 3))

        # 重新排列像素
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        i = self.ConvTranspose(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        output = x + i
        return output
