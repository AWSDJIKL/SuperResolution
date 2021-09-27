# -*- coding: utf-8 -*-
'''
复现亚像素卷积的效果
'''
# @Time    : 2021/7/14 16:28
# @Author  : LINYANZHEN
# @File    : train.py
import random
import torchvision.models
from torch.optim.lr_scheduler import MultiStepLR
import os
import model
import time
import utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def train_and_val(model, train_loader, val_loader, criterion, optimizer, epoch):
    psnr_list = []
    loss_list = []
    best_psnr = 0
    for i in range(epoch):
        epoch_psnr = 0
        epoch_loss = 0
        count = 0
        epoch_start_time = time.time()
        print("epoch[{}/{}]".format(i, epoch))
        model.train()
        for index, (x, y) in enumerate(train_loader, 0):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            count += len(x)
        epoch_loss /= count
        count = 0
        model.eval()
        for index, (x, y) in enumerate(val_loader, 0):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            epoch_psnr += utils.calculate_psnr(y, out)
            count += len(x)
        epoch_psnr /= count

        psnr_list.append(epoch_psnr)
        loss_list.append(epoch_loss)
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            # 保存最优模型
            torch.save(model.state_dict(), "sub_pixel_convolution.pth")
            print("模型已保存")

        print("psnr:{}  best psnr:{}".format(epoch_psnr, best_psnr))
        print("loss:{}".format(epoch_loss))
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(psnr_list, 'b', label='psnr')
    plt.legend()
    plt.grid()
    plt.title('best psnr=%5.2f' % best_psnr)
    plt.savefig('psnr.jpg', dpi=256)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, 'r', label='loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss.jpg', dpi=256)
    plt.close()


if __name__ == '__main__':
    start_time = time.time()

    device = torch.device('cuda:0')
    train_loader, val_loader = utils.get_super_resolution_dataloader("Aircraft_ycbcr")
    # cudnn.benchmark = True
    model = model.SPCNet(upscale_factor=3)
    model = model.to(device)

    lr = 1e-3
    epoch = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 分模块调整学习率暂时作用不大
    # optimizer = torch.optim.Adam([
    #     {'params': model.first_part.parameters()},
    #     {'params': model.last_part.parameters(), 'lr': lr * 10}
    # ], lr=lr)
    # 调整学习率，在第40，80个epoch时改变学习率
    scheduler = MultiStepLR(optimizer, milestones=[int(epoch * 0.4), int(epoch * 0.8)], gamma=0.1)
    criterion = nn.MSELoss()
    # 训练模型
    train_and_val(model, train_loader, val_loader, criterion, optimizer, epoch)
    # 保存模型
    print("训练完成")
    print("总耗时:{}min".format((time.time() - start_time) / 60))
