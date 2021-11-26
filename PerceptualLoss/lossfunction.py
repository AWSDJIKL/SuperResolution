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
        self.PerceptualModel = PerceptualVGG16(output_layer).cuda()
        self.PerceptualModel.eval()

    def forward(self, pred, y):
        # pred = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(pred)
        # y = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(y)
        # 定义计算过程
        # 输出结果与真实结果一起经过感知损失模型
        perceptual_pred = self.PerceptualModel(pred)
        perceptual_y = self.PerceptualModel(y)

        (batch_size, c, h, w) = perceptual_y.size()
        perceptual_loss = torch.nn.MSELoss()(perceptual_pred, perceptual_y) / (batch_size * c * h * w)
        # mse_loss = torch.nn.MSELoss()(pred, y)
        # loss = 0.25 * mse_loss + 0.75 * perceptual_loss
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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation=None,
            norm_layer=None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class PerceptualResNet50(nn.Module):
    def __init__(self):
        super(PerceptualResNet50, self).__init__()
        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        pretrain_state_dict = models.resnet50(pretrained=True, progress=True).state_dict()
        # print(pretrain_state_dict)
        # print(self.state_dict())
        # 加载预训练权重
        self.resnet.load_state_dict(pretrain_state_dict, False)
        # print(self.state_dict())
        # 不计算梯度
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        return x


class resnet_loss(nn.Module):
    def __init__(self):
        super(resnet_loss, self).__init__()
        self.PerceptualModel = PerceptualResNet50().cuda()
        self.PerceptualModel.eval()

    def forward(self, pred, y):
        # pred = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(pred)
        # y = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(y)
        # 定义计算过程
        # 输出结果与真实结果一起经过感知损失模型
        perceptual_pred = self.PerceptualModel(pred)
        perceptual_y = self.PerceptualModel(y)

        (batch_size, c, h, w) = perceptual_y.size()
        perceptual_loss = torch.nn.MSELoss()(perceptual_pred, perceptual_y) / (batch_size * c * h * w)
        # mse_loss = torch.nn.MSELoss()(pred, y)
        # loss = 0.25 * mse_loss + 0.75 * perceptual_loss
        loss = perceptual_loss
        pred.cpu()
        y.cpu()
        torch.cuda.empty_cache()

        return loss


if __name__ == '__main__':
    loss_model = PerceptualResNet50()
