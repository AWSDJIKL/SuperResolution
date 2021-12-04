# -*- coding: utf-8 -*-
'''
常规图片训练集训练
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : train.py
import argparse
import datetime
from PIL import Image
import os
import random
import shutil
import time
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import utils
from PerceptualLoss import lossfunction
# from SubPixelConvolution import model
from JohnsonSR import model
from utils import calculate_psnr  # noqa: E402
import datasets
import config
import tqdm


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
        progress = tqdm.tqdm(train_loader, total=len(train_loader))
        for (x, y) in progress:
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
            # progress = tqdm.tqdm(val_loader, total=len(train_loader))
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
    # parser.add_argument("--experiment_name", default="SPC_with_PL", type=str, help="experiment name")
    # parser.add_argument("--use_pl", default=True, type=bool, help="use Perceptual Loss")
    parser.add_argument("--use_pl", default=True, type=lambda x: x.lower() == 'true', help="use Perceptual Loss")
    vgg16_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
    parser.add_argument("--output_layer", default="relu2_2", type=str, choices=vgg16_layers,
                        help="Perceptual Loss's output layer")
    parser.add_argument("--use_pretrain", default=False, type=lambda x: x.lower() == 'true', help="use Pretrain model")
    parser.add_argument("--model_path",
                        default="checkpoint/GradualSR_without_PL_x4/GradualSR_without_PL_x4_final_epoch.pth",
                        type=str, help="pretrain model path")

    start_time = time.time()
    # 固定随机种子
    # setup_seed(100)
    args = parser.parse_args()
    # torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda:0')
    train_loader, val_loader = datasets.get_super_resolution_dataloader(args)

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # model = model.SPCNet(args.upscale_factor)
    # model = model.Residual_SPC(args.upscale_factor)
    # model = model.GradualSR(args.upscale_factor)
    model = model.JohnsonSR()
    model = model.to(device)

    if args.use_pretrain:
        print("加载预训练模型（在此基础上继续训练）")
        model.load_state_dict(torch.load(args.model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 调整学习率，在第40，80个epoch时改变学习率
    scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch * 0.8)], gamma=0.1)
    if args.use_pl:
        criterion = lossfunction.vgg16_loss(output_layer=args.output_layer)
        # criterion = lossfunction.resnet_loss()
        # experiment_name = model.__class__.__name__ + "_with_mix_PL_" + args.output_layer + "_x" + str(args.upscale_factor)
        experiment_name = model.__class__.__name__ + "_with_PL_" + args.output_layer + "_x" + str(args.upscale_factor)
    else:
        criterion = nn.MSELoss()
        experiment_name = model.__class__.__name__ + "_without_PL_x" + str(args.upscale_factor)
    # 训练模型
    train_and_val(model, train_loader, val_loader, criterion, optimizer, args.epoch, experiment_name)

    # 保存模型
    print("训练完成")
    print("总耗时:" + utils.time_format(time.time() - start_time))
