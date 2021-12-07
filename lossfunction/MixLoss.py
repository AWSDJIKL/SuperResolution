# -*- coding: utf-8 -*-
'''
混合损失
'''
# @Time    : 2021/12/5 10:33
# @Author  : LINYANZHEN
# @File    : MixLoss.py
import torch.nn
from lossfunction import PerceptualLoss
from lossfunction import TestureLoss
# import PerceptualLoss
# import TestureLoss

import torch.nn as nn


class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.perceptual_loss = PerceptualLoss.vgg16_loss()
        self.texture_loss = TestureLoss.StyleLoss(weight=5)

    def forward(self, pred, y):
        mse_loss = self.mse_loss(pred, y)
        perceptual_loss = self.perceptual_loss(pred, y)
        texture_loss = self.texture_loss(pred, y)
        loss = mse_loss + perceptual_loss + texture_loss
        return loss
