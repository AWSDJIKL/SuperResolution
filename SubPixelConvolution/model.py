# -*- coding: utf-8 -*-
'''
亚像素卷积模型
'''
# @Time    : 2021/7/14 16:26
# @Author  : LINYANZHEN
# @File    : model.py
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import config
from torch.nn.modules.utils import _pair, _quadruple


class SPCNet(nn.Module):
    def __init__(self, upscale_factor, in_channels=3):
        '''
        亚像素卷积网络

        :param upscale_factor: 放大倍数
        '''
        super(SPCNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, in_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        # 重新排列像素
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        # x = self.tanh(x)
        # x = torch.add(x, 1.)
        # x = torch.mul(x, 0.5)
        return x


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
    weighted_pix = F.conv2d(input_pad, weight=kernel, stride=1, padding=0, groups=c)
    return weighted_pix


class WashGrad(torch.autograd.Function):
    '''
    洗掉梯度中的振铃效应
    '''

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
    def backward(ctx, grad):
        '''

        :param ctx:
        :param grad_output: 链式法则上一层的梯度
        :return:
        '''
        if config.wash_grad:
            # 用高斯滤波将梯度中的振铃效应洗掉
            grad = gaussian_blur(grad, kernel_size=3)
        # 归一化
        grad = grad / (grad.max() - grad.min())
        # 该层梯度为1
        return grad * 1


class PrintGrad(torch.autograd.Function):
    '''
    将梯度转化为图像
    '''

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
        if config.print_grad:
            channel_count = 0
            # 将梯度tensor转化为图像输出
            grad = grad_output.squeeze(0)
            # print(grad.size())
            # 分开每个通道都输出一张图片
            for i in range(grad.size()[0]):
                # print(grad[i, :, :])
                g = grad[i, :, :]
                # g = g / (g.max() - g.min())
                img = transforms.ToPILImage()(g)
                if config.wash_grad:
                    img.save("wash_grad/{}.jpg".format(channel_count))
                else:
                    img.save("no_wash_grad/{}.jpg".format(channel_count))
                channel_count += 1
        # 该层梯度为1
        return grad_output * 1


class NoiseGenerator(nn.Module):
    def __init__(self):
        '''
        生成噪音

        '''
        super(NoiseGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        '''
        残差块

        :param input_channels: 输入通道数
        :param output_channels: 输出通道数
        '''
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, (3, 3), (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, (3, 3), (1, 1), (1, 1))
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv1x1 = nn.Conv2d(input_channels, output_channels, (1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        output = self.conv1(x)
        # output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        # output = self.bn2(output)
        # output = self.relu(output)

        # identity = self.conv1x1(identity)
        output = output + identity
        # output = self.relu(output)
        return output


class Residual_SPC(nn.Module):
    def __init__(self, upscale_factor, in_channels=3):
        '''
        亚像素卷积网络(使用残差块优化网络)

        :param upscale_factor: 放大倍数
        '''
        super(Residual_SPC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))
        self.block1 = ResidualBlock(input_channels=64, output_channels=64)
        self.block2 = ResidualBlock(input_channels=64, output_channels=64)
        self.block3 = ResidualBlock(input_channels=64, output_channels=64)
        self.block4 = ResidualBlock(input_channels=64, output_channels=64)
        self.conv2 = nn.Conv2d(64, in_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))

        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=upscale_factor)
        # self.conv1x1 = nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1))
        # 重新排列像素
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # identity = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.conv2(x)
        x = self.pixel_shuffle(x)

        output = x
        # identity = self.upsample(identity)
        # # identity = self.conv1x1(identity)
        # output = x + identity

        return output


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(ResBlock, self).__init__()
        # self.padding = nn.ReflectionPad2d((1, 1, 1, 1))
        # padding = (0, 0)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation)
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_in = x
        # x = self.padding(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.padding(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x + x_in
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(UpBlock, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        # self.upsample = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), (2, 2))
        # self.batchnorm = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, x):

        x = self.upsample(x)
        x = self.conv(x)
        # x = self.batchnorm(x)
        x = self.relu(x)
        return x


class GradualSR(nn.Module):
    def __init__(self, upscale_factor):
        super(GradualSR, self).__init__()
        # self.padding = nn.ReflectionPad2d(4)
        self.conv_in = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), (1, 1))
        self.resblock1 = ResBlock(64, 64, (3, 3), (1, 1), (1, 1), (1, 1))
        self.resblock2 = ResBlock(64, 64, (3, 3), (1, 1), (1, 1), (1, 1))
        self.resblock3 = ResBlock(64, 64, (3, 3), (1, 1), (1, 1), (1, 1))
        self.resblock4 = ResBlock(64, 64, (3, 3), (1, 1), (1, 1), (1, 1))
        self.upsample1 = UpBlock(64, 64, (3, 3), (1, 1))
        self.upsample2 = UpBlock(64, 64, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        # self.pool = nn.MaxPool2d(2)
        # self.pixel_shffule = nn.PixelShuffle(upscale_factor)
        # self.upsample = UpBlock(64, 64, upscale_factor, 3, 1, 1)
        # self.medfilt = MedianPool2d(3)
        # self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.bicubic = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        x_in = x
        # x = self.padding(x)
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        # x = self.padding(x)
        # x = self.conv_out(x)
        # x = self.medfilt(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        x_in = self.bicubic(x_in)
        x = x + x_in
        return x


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=1, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return x


