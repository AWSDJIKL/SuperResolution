# -*- coding: utf-8 -*-
'''
纹理损失
'''
# @Time    : 2021/12/5 10:27
# @Author  : LINYANZHEN
# @File    : TestureLoss.py
import torch
import torch.nn as nn


class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class GramMatrix(nn.Module):

    def forward(self, x, patch_size):
        b, c, w, h = x.shape
        # w = x.shape[2]
        # h = x.shape[3]
        x = x.reshape((x.shape[0], x.shape[1], int(w / patch_size), patch_size, int(h / patch_size), patch_size))
        x = x.transpose(1, 3).transpose(2, 5).transpose(4, 5)
        x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5])  # b*p*p,c,h/p,w/p
        x = x.transpose(0, 1)  # c,b*p*p,h/p,w/p
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # G = torch.mm(x, x.transpose(1, 2))  # compute the gram product
        G = torch.matmul(x, x.transpose(1, 2))
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * w * h)


def crop_feature(x, patch_size):
    # input:b,c,h,w
    w = x.shape[2]
    h = x.shape[3]
    x = x.view((x.shape[0], x.shape[1], int(w / patch_size), patch_size, int(h / patch_size), patch_size))
    x = x.transpose(1, 3).transpose(2, 5).transpose(4, 5)
    x = x.view(-1, x.shape[3], x.shape[4], x.shape[5])  # b*p*p,c,h/p,w/p
    x = x.transpose(0, 1)  # c,b*p*p,h/p,w/p
    x = x.view(x.shape[0], x.shape[1], -1)
    return x


class StyleLoss(nn.Module):

    def __init__(self, weight, patch_size=16):
        super(StyleLoss, self).__init__()
        # self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
        self.patch_size = patch_size

    def forward(self, input, target):
        # target = target.detach() * self.weight
        input = self.gram(input, self.patch_size)
        target = self.gram(target, self.patch_size)
        loss = self.criterion(input, target)
        return loss

    # def backward(self, retain_graph=True):
    #     self.loss.backward(retain_graph=retain_graph)
    #     return self.loss
