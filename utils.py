# -*- coding: utf-8 -*-
'''
工具函数
'''
# @Time    : 2021/8/5 15:08
# @Author  : LINYANZHEN
# @File    : utils.py
import math
import os

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import torch.nn.functional as F


def calculate_psnr(img1, img2):
    '''
    计算两张图片之间的PSNR误差

    :param img1:
    :param img2:
    :return:
    '''
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2)).item()


def create_kernel(kernel_size, channel):
    '''
    创建计算核并分配权重
    :param kernel_size:
    :return:
    '''
    # 仅计算平均数
    kernel = torch.Tensor([[
        [[1 for i in range(kernel_size)] for i in range(kernel_size)]
        for i in range(channel)] for i in range(channel)]).cuda()
    kernel /= kernel.sum()
    return kernel


def calculate_ssim(img1, img2, kernel_size=11):
    '''

    :param img1:
    :param img2:
    :param kernel_size: 滑动窗口大小
    :return:
    '''
    k1 = 0.01
    k2 = 0.03
    if torch.max(img1) > 128:
        max = 255
    else:
        max = 1
    if torch.min(img1) < -0.5:
        min = -1
    else:
        min = 0
    l = max - min
    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2
    (channel, h, w) = img1.size()
    kernel = create_kernel(kernel_size, channel)
    # print(kernel)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    # 计算均值
    mean1 = F.conv2d(img1, weight=kernel, stride=1, padding=0)
    mean2 = F.conv2d(img2, weight=kernel, stride=1, padding=0)
    # print(img1.size())
    # print(mean1.size())
    # 计算方差,利用公式dx=e(x^2)-e(x)^2
    variance1 = F.conv2d(img1 ** 2, weight=kernel, stride=1, padding=0) - mean1 ** 2
    variance2 = F.conv2d(img2 ** 2, weight=kernel, stride=1, padding=0) - mean2 ** 2
    # 计算协方差
    covariance = F.conv2d(img1 * img2, weight=kernel, stride=1, padding=0) - (mean1 * mean2)

    ssim = torch.mean(((2 * mean1 * mean2 + c1) * (2 * covariance + c2)) / (
            (mean1 ** 2 + mean2 ** 2 + c1) * (variance1 + variance2 + c2)))
    return ssim


def calculate_LPIPS(img1, img2):
    '''

    :param img1:
    :param img2:
    :return:
    '''


def load_image_RGB(image_path):
    '''
    正常加载图片

    :param image_path: 图片路径
    :return:
    '''
    image = Image.open(image_path).convert("RGB")
    return image


def load_image_ycbcr(image_path):
    '''
    以YCbCr格式加载图片并按通道分割

    :param image_path: 图片路径
    :return:
    '''
    y, cb, cr = Image.open(image_path).convert('YCbCr').split()
    # y = Variable(ToTensor()(y))
    # cb = Variable(ToTensor()(cb))
    # cr = Variable(ToTensor()(cr))
    return y, cb, cr


def lr_transform(img_size, upscale_factor):
    '''

    :param img_size: 原图大小
    :param upscale_factor:
    :return:
    '''
    new_size = [i // upscale_factor for i in img_size]
    # print("original size =", img_size)
    # print("new size =", new_size)
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.Resize(new_size),
        transforms.ToTensor(),
    ])


def hr_transform(img_size):
    '''

    :param img_size:
    :return:
    '''
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def tensor_to_image(tensor):
    return transforms.ToPILImage()(tensor)


def time_format(second):
    m, s = divmod(second, 60)
    m = round(m)
    s = round(s)
    if m < 60:
        return "{}m{}s".format(m, s)
    else:
        h, m = divmod(m, 60)
        h = round(h)
        m = round(m)
    if h < 24:
        return "{}h{}m{}s".format(h, m, s)
    else:
        d, h = divmod(h, 24)
        d = round(d)
        h = round(h)
    return "{}d{}h{}m{}s".format(d, h, m, s)


def test_model(model, test_image_path, upscale_factor, save_name):
    '''
    测试模型效果

    :param model: 要测试的模型
    :param test_image_path: 用于测试的图片的位置
    :param upscale_factor: 放大倍数
    :return:
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    origin_image = Image.open(test_image_path).convert('RGB')
    img_name, suffix = os.path.splitext(test_image_path)
    image_width = (origin_image.width // upscale_factor) * upscale_factor
    image_height = (origin_image.height // upscale_factor) * upscale_factor

    # hr_image = origin_image.resize((image_width, image_height), resample=Image.BICUBIC)
    # lr_image = origin_image.resize((image_width // upscale_factor, image_height // upscale_factor),resample=Image.BICUBIC)
    lr_image = lr_transform((image_height, image_width), upscale_factor)(origin_image)
    hr_image = hr_transform((image_height, image_width))(origin_image)

    # bicubic = lr_image.resize((image_width, image_height), resample=Image.BICUBIC)
    # psnr = calculate_psnr(Variable(ToTensor()(hr_image)).to(device),
    #                       Variable(ToTensor()(bicubic)).to(device))
    # ssim = calculate_ssim(Variable(ToTensor()(hr_image)).to(device),
    #                       Variable(ToTensor()(bicubic)).to(device))
    # bicubic.save(img_name + "_bicubic_x{}".format(upscale_factor) + suffix)
    bicubic = transforms.Resize((image_height, image_width))(lr_image)
    psnr = calculate_psnr(hr_image.to(device), bicubic.to(device))
    ssim = calculate_ssim(hr_image.to(device), bicubic.to(device))
    transforms.ToPILImage()(bicubic).convert('RGB').save(img_name + "_bicubic_x{}".format(upscale_factor) + suffix)

    print('bicubic PSNR: {}'.format(psnr))
    print("bicubic SSIM: {}".format(ssim))

    # x = Variable(ToTensor()(lr_image)).to(device).unsqueeze(0)  # 补上batch_size那一维
    # y = Variable(ToTensor()(hr_image)).to(device)
    x = lr_image.to(device).unsqueeze(0)  # 补上batch_size那一维
    y = hr_image.to(device)

    with torch.no_grad():
        out = model(x).clip(0, 1).squeeze()
        # out = model(x).squeeze()
    psnr = calculate_psnr(y, out)
    ssim = calculate_ssim(y, out)
    print('{} PSNR: {}'.format(save_name, psnr))
    print('{} SSIM: {}'.format(save_name, ssim))
    out = tensor_to_image(out)
    out.save(img_name + '_{}_x{}'.format(save_name, upscale_factor) + suffix)
    return
