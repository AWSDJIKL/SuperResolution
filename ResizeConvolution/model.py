# -*- coding: utf-8 -*-
'''
缩放卷积，先用双线性插值将原图放大，然后用卷积层学习细节
'''
# @Time    : 2021/8/16 17:11
# @Author  : LINYANZHEN
# @File    : model.py
import torch
import torch.nn as nn
import typing
import numpy as np
import torch.nn.functional as F


def get_gaussian_kernel(kernel_size, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    center = kernel_size // 2
    xs = np.arange(kernel_size, dtype=np.float32) - center
    # 一维卷积核
    kernel_1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel_1d[..., None] @ kernel_1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur(input, kernel_size, sigma=0):
    kernel = get_gaussian_kernel(kernel_size, sigma)
    b, c, h, w = input.shape
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    pad = (kernel_size - 1) // 2
    input_pad = F.pad(input, pad=[pad, pad, pad, pad], mode="reflect")
    input_pad = input_pad.cuda()
    kernel = kernel.cuda()
    weighted_pix = F.conv2d(input_pad, weight=kernel, bias=None, stride=1, padding=0, groups=c)
    return weighted_pix


class WashGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        '''
        什么都不做

        :param ctx:
        :param i: 输入
        :return:
        '''
        i = i * 1
        ctx.save_for_backward(i)
        return i

    @staticmethod
    def backward(ctx, grad_output):
        '''

        :param ctx:
        :param grad_output: 链式法则上一层的梯度
        :return:
        '''
        # 用高斯滤波将梯度中的振铃效应洗掉
        grad_input = gaussian_blur(grad_output, 3)
        # 该层梯度为1
        return grad_input * 1


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
        self.wash_grad = WashGrad

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode="bilinear", align_corners=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = torch.sinc(x)
        x = self.wash_grad.apply(x)
        x = self.conv2(x)
        return x
