# -*- coding: utf-8 -*-
'''
自定义损失函数
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : lossfunction.py
import torch
import torch.nn as nn
import torch.nn.functional as func
import model


class PerceptualLoss(nn.Module):
    '''
    感知损失
    '''

    def __init__(self):
        # 定义必要的超参数
        super(PerceptualLoss, self).__init__()
        self.PerceptualModel = model.PerceptualVGG16("relu2_2").cuda()
        self.PerceptualModel.eval()

    def forward(self, pred, y):
        # 定义计算过程
        # 输出结果与真实结果一起经过感知损失模型
        pred = self.PerceptualModel(pred)
        y = self.PerceptualModel(y)
        # 获取作为label的来自于预训练的深度神经网络中间层的输出的大小
        # print(pred.size())
        # print(y.size())
        (batch_size, c, h, w) = y.size()
        # loss = (torch.sum((pred[0] - y[0]) ** 2) ** 0.5) / (c * h * w)
        # # 分开计算batch内每个样本的损失
        # for i in range(1, batch_size):
        #     loss += (torch.sum((pred[i] - y[i]) ** 2) ** 0.5) / (c * h * w)
        loss = (torch.sum((pred - y) ** 2) ** 0.5) / (batch_size * c * h * w)
        pred.cpu()
        y.cpu()
        torch.cuda.empty_cache()
        return loss
