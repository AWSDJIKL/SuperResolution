# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/4 11:29
# @Author  : LINYANZHEN
# @File    : model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvPrelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.conv(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super(ResBlock, self).__init__()
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_in = x
        x = self.conv3x3_1(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.conv3x3_2(x)
        x = self.batchnorm(x)
        return x + x_in


class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvBatchRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.batchnorm(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        return F.relu(x)


class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        self.conv1 = ConvPrelu(3, 56, kernel_size=5)  # feature extraction
        self.conv2 = ConvPrelu(56, 12, kernel_size=1)  # strinking
        self.conv3 = ConvPrelu(12, 12, kernel_size=3, padding=1)  # feature extraction x4
        self.conv4 = ConvPrelu(12, 12, kernel_size=3, padding=1)
        self.conv5 = ConvPrelu(12, 12, kernel_size=3, padding=1)
        self.conv6 = ConvPrelu(12, 12, kernel_size=3, padding=1)
        self.conv7 = ConvPrelu(12, 56, kernel_size=1)  # shrinking
        self.deconv = nn.ConvTranspose2d(56, 3, kernel_size=9, stride=3, padding=4)  # upsample

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.deconv(x)
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

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv_in(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.conv_out(x)
        return F.tanh(x)
