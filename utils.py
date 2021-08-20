# -*- coding: utf-8 -*-
'''
工具函数
'''
# @Time    : 2021/8/5 15:08
# @Author  : LINYANZHEN
# @File    : utils.py
import os

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
from torchvision.transforms import ToTensor


def calaculate_psnr(img1, img2):
    '''
    计算两张图片之间的PSNR误差

    :param img1:
    :param img2:
    :return:
    '''
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2)).item()


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
        transforms.Resize(new_size, interpolation=InterpolationMode.BICUBIC),
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
    image_width = (origin_image.width // upscale_factor) * upscale_factor
    image_height = (origin_image.height // upscale_factor) * upscale_factor
    hr_image = origin_image.resize((image_width, image_height), resample=Image.BICUBIC)
    lr_image = origin_image.resize((image_width // upscale_factor, image_height // upscale_factor),
                                   resample=Image.BICUBIC)
    img_name, suffix = os.path.splitext(test_image_path)
    # print(img_name)
    # print(suffix)
    bicubic = lr_image.resize((image_width, image_height), resample=Image.BICUBIC)
    psnr = calaculate_psnr(Variable(ToTensor()(hr_image)).to(device),
                           Variable(ToTensor()(bicubic)).to(device))

    print('bicubic PSNR: {}'.format(psnr))
    bicubic.save(img_name + "_bicubic_x{}".format(upscale_factor) + suffix)

    x = Variable(ToTensor()(lr_image)).to(device).unsqueeze(0)
    y = Variable(ToTensor()(hr_image)).to(device)
    with torch.no_grad():
        out = model(x).clip(0, 1).squeeze()
        # out = model(x).squeeze()
    psnr = calaculate_psnr(y, out)
    print('{} PSNR: {}'.format(save_name, psnr))
    out = tensor_to_image(out)
    out.save(img_name + '_{}_x{}'.format(save_name, upscale_factor) + suffix)
    return
