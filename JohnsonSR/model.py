# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/4 11:29
# @Author  : LINYANZHEN
# @File    : model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super(ResBlock, self).__init__()
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.conv3x3_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)
        x = self.batchnorm(x)
        return x + x_in


class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvBatchRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.batchnorm(x)
        x = self.relu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        # x = self.batchnorm(x)
        x = self.relu(x)
        return x


class JohnsonSR(nn.Module):
    def __init__(self):
        super(JohnsonSR, self).__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.conv_in = ConvBatchRelu(3, 64, kernel_size=9, padding=4)
        self.resblock1 = ResBlock(64, 64, padding=1)
        self.resblock2 = ResBlock(64, 64, padding=1)
        self.resblock3 = ResBlock(64, 64, padding=1)
        self.resblock4 = ResBlock(64, 64, padding=1)
        self.upblock1 = UpBlock(64, 64, kernel_size=3, padding=1)
        self.upblock2 = UpBlock(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x = self.batch_norm(x)
        x = self.conv_in(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.conv_out(x)
        # x = self.tanh(x)
        return x
