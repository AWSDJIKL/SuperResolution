# -*- coding: utf-8 -*-
'''
训练
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : train.py
import argparse
import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR

import lossfunction
import model

# 下面是需要从上一级目录开始导入的模块
sys.path.append(os.getcwd())
# print(os.getcwd())
# print(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils import calaculate_psnr  # noqa: E402
import datasets


def train_and_val(model, train_loader, val_loader, criterion, optimizer, epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
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
            # print(x.size())
            # print(out.size())
            # print(y.size())
            loss = criterion(out, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            count += len(x)
            x = x.cpu()
            y = y.cpu()
            out = out.cpu()
            loss.cpu()
            torch.cuda.empty_cache()
        epoch_loss /= count
        count = 0
        model.eval()
        with torch.no_grad():
            for index, (x, y) in enumerate(val_loader, 0):
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                epoch_psnr += calaculate_psnr(y, out)
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
            # 保存psnr最优模型
            torch.save(model.state_dict(),
                       "../checkpoint/PerceptualLoss_{}_{}_{}_best.pth".format(datetime.date.year,
                                                                               datetime.date.month, datetime.date.day))
            print("模型已保存")

        print("psnr:{}  best psnr:{}".format(epoch_psnr, best_psnr))
        print("loss:{}".format(epoch_loss))
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))

    # 保存最后一个epoch的模型，作为比对
    torch.save(model.state_dict(),
               "../checkpoint/PerceptualLoss_{}_{}_{}_final_epoch.pth".format(datetime.date.year,
                                                                              datetime.date.month, datetime.date.day))
    print("模型已保存")

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
    parser = argparse.ArgumentParser(description="感知损失模型")
    parser.add_argument("--upscale_factor", default=3, type=int, help="scale factor, Default: 3")
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--num_workers", default=1, type=int, help="num_workers")
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--epoch", default=100, type=int, help="epoch")

    start_time = time.time()

    args = parser.parse_args()
    device = torch.device('cuda:0')
    train_loader, val_loader = datasets.get_super_resolution_dataloader(args)
    cudnn.benchmark = True
    model = model.Sub_pixel_conv(args.upscale_factor)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 调整学习率，在第40，80个epoch时改变学习率
    scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch * 0.4), int(args.epoch * 0.8)], gamma=0.1)
    # criterion = nn.MSELoss()
    criterion = lossfunction.vgg16_loss()
    # 训练模型
    train_and_val(model, train_loader, val_loader, criterion, optimizer, args.epoch)
    # 保存模型
    print("训练完成")
    print("总耗时:{}min".format((time.time() - start_time) / 60))
