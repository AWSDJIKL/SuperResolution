# -*- coding: utf-8 -*-
'''
训练
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : train.py
import random
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import os
import model
import time
import utils
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import lossfunction


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
            # print(out.size())
            # print(y.size())

            loss = criterion(out, y)
            loss.backward()
            epoch_loss += loss.item()
            # print(epoch_loss)
            optimizer.step()
            count += len(x)
            x = x.cpu()
            y = y.cpu()
            out = out.cpu()
            loss.cpu()
            torch.cuda.empty_cache()
            # print(index)
        epoch_loss /= count
        count = 0
        model.eval()
        with torch.no_grad():
            for index, (x, y) in enumerate(val_loader, 0):
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                epoch_psnr += utils.calaculate_psnr(y, out)
                count += len(x)
                x = x.cpu()
                y = y.cpu()
                out = out.cpu()
                torch.cuda.empty_cache()
        epoch_psnr /= count

        psnr_list.append(epoch_psnr)
        loss_list.append(epoch_loss)
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            # 保存最优模型
            torch.save(model, "sub_pixel_convolution.pth")
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
    train_loader, test_loader = utils.get_super_resolution_dataloader("Aircraft")
    # cudnn.benchmark = True
    model = model.Sub_pixel_conv(upscale_factor=3)
    model = model.to(device)

    lr = 1e-3
    epoch = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 调整学习率，在第40，80个epoch时改变学习率
    scheduler = MultiStepLR(optimizer, milestones=[int(epoch * 0.4), int(epoch * 0.8)], gamma=0.1)
    # criterion = nn.MSELoss()
    criterion = lossfunction.PerceptualLoss()
    # 训练模型
    train_and_val(model, train_loader, test_loader, criterion, optimizer, epoch)
    # 保存模型
    print("训练完成")
    print("总耗时:{}min".format((time.time() - start_time) / 60))
