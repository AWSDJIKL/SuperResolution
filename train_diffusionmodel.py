# -*- coding: utf-8 -*-
'''
训练
'''
# @Time    : 2021/11/3 14:47
# @Author  : LINYANZHEN
# @File    : train_diffusionmodel.py
import argparse
import datetime
from PIL import Image
import os
import random
import shutil
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.backends import cudnn
import numpy as np
import utils
from utils import calculate_psnr  # noqa: E402
import datasets
import config

from DiffusionModel import model


def train_and_val(model, train_loader, val_loader, criterion, optimizer, epoch, experiment_name):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    save_path = os.path.join("checkpoint", experiment_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    test_img_path = os.path.join(save_path, "test.png")
    Image.open("img_test/test.png").save(test_img_path)
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
            # if index == (len(train_loader) - 1) and i == (epoch - 1):
            #     config.print_grad = True
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
            # if config.print_grad:
            #     torchvision.transforms.ToPILImage()(y.squeeze(0)).save("wash_grad/y.jpg")
            #     torchvision.transforms.ToPILImage()(y.squeeze(0)).save("no_wash_grad/y.jpg")
            # return

        epoch_loss /= count
        count = 0
        model.eval()
        with torch.no_grad():
            for index, (x, y) in enumerate(val_loader, 0):
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                epoch_psnr += calculate_psnr(y, out)
                count += len(x)
                x = x.cpu()
                y = y.cpu()
                out = out.cpu()
                torch.cuda.empty_cache()
        epoch_psnr /= count
        psnr_list.append(epoch_psnr)
        loss_list.append(epoch_loss)

        save_name = "{}".format(i)
        utils.test_model(model, test_img_path, args.upscale_factor, save_name)
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            # 保存psnr最优模型
            torch.save(model.state_dict(),
                       os.path.join(save_path, "{}_best.pth".format(experiment_name)))
            print("模型已保存")

        print("psnr:{}  best psnr:{}".format(epoch_psnr, best_psnr))
        print("loss:{}".format(epoch_loss))
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))

    # 保存最后一个epoch的模型，作为比对
    torch.save(model.state_dict(),
               os.path.join(save_path, "{}_final_epoch.pth".format(experiment_name)))
    print("模型已保存")

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(psnr_list, 'b', label='psnr')
    plt.legend()
    plt.grid()
    plt.title('best psnr=%5.2f' % best_psnr)
    plt.savefig(os.path.join(save_path, 'psnr.jpg'), dpi=256)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, 'r', label='loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, 'loss.jpg'), dpi=256)
    plt.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="感知损失模型")
    parser.add_argument("--upscale_factor", default=4, type=int, help="scale factor, Default: 3")
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--epoch", default=100, type=int, help="epoch")

    start_time = time.time()
    # 固定随机种子
    setup_seed(100)
    args = parser.parse_args()
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda:0')
    train_loader, val_loader = datasets.get_super_resolution_dataloader(args)

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    model = model.DenoiseModel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 调整学习率，在第40，80个epoch时改变学习率
    # scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch * 0.8)], gamma=0.1)
    criterion = nn.MSELoss()
    experiment_name = model.__class__.__name__ + "_without_PL_x" + str(args.upscale_factor)
    # 训练模型
    train_and_val(model, train_loader, val_loader, criterion, optimizer, args.epoch, experiment_name)
    # 保存模型
    print("训练完成")
    print("总耗时:" + utils.time_format(time.time() - start_time))
