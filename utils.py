# -*- coding: utf-8 -*-
'''
工具函数
'''
# @Time    : 2021/8/5 15:08
# @Author  : LINYANZHEN
# @File    : utils.py
import math
import os

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import SubPixelConvolution.model


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


class AddGaussianNoise():
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype("uint8")).convert("RGB")
        return img


def lr_transform(img_size, upscale_factor):
    '''

    :param img_size: 原图大小
    :param upscale_factor:
    :return:
    '''
    new_size = [i // upscale_factor for i in img_size]
    # vgg_mean = torch.tensor((103.939, 116.779, 123.68)).view(3, 1, 1).expand((3, new_size[0], new_size[1]))
    # vgg_mean = torch.tensor((123.68, 116.779, 103.939)).view(3, 1, 1).expand((3, new_size[0], new_size[1]))
    # print("original size =", img_size)
    # print("new size =", new_size)
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.Resize(new_size),
        transforms.ToTensor(),
        # transforms.Lambda(lambda img: img[torch.LongTensor([2, 1, 0])]),
        # transforms.Lambda(lambda img: img * 255 - vgg_mean),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def hr_transform(img_size):
    '''

    :param img_size:
    :return:
    '''
    # vgg_mean = torch.tensor((103.939, 116.779, 123.68)).view(3, 1, 1).expand((3, img_size[0], img_size[1]))
    # vgg_mean = torch.tensor((123.68, 116.779, 103.939)).view(3, 1, 1).expand((3, img_size[0], img_size[1]))
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        # transforms.Lambda(lambda img: img[torch.LongTensor([2, 1, 0])]),
        # transforms.Lambda(lambda img: img * 255 - vgg_mean)
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def deprocess(img_size):
    vgg_mean = torch.tensor((123.68, 116.779, 103.939)).view(3, 1, 1).expand((3, img_size[0], img_size[1])).cuda()
    return transforms.Compose([
        # transforms.Lambda(lambda img: img[torch.LongTensor([2, 1, 0])]),
        transforms.Lambda(lambda img: (img + vgg_mean) / 255)
    ])


def tensor_to_image(tensor):
    # tr_mean, tr_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # mu = torch.Tensor(tr_mean).view(-1, 1, 1).cuda()
    # sigma = torch.Tensor(tr_std).view(-1, 1, 1).cuda()
    # img = transforms.ToPILImage()((tensor * sigma + mu).clamp(0, 1))
    img = transforms.ToPILImage()(tensor)
    return img


def histogram_match(input, target, patch=3, stride=3):
    '''
    直方图匹配

    :param input: 需要匹配的图
    :param target: 匹配的对象
    :param patch: 直方图bin分割数
    :param stride: 移动步长
    :return:
    '''
    c1, h1, w1 = input.size()
    c2, h2, w2 = target.size()
    input.resize_(h1 * w1 * h2 * w2)
    target.resize_(h2 * w2 * h2 * w2)
    conv = torch.tensor((), dtype=torch.float32)
    conv = conv.new_zeros((h1 * w1, h2 * w2))
    conv.resize_(h1 * w1 * h2 * w2)
    assert c1 == c2, 'input:c{} is not equal to target:c{}'.format(c1, c2)

    size1 = h1 * w1
    size2 = h2 * w2
    N = h1 * w1 * h2 * w2
    print('N is', N)

    for i in range(0, N):
        i1 = i / size2
        i2 = i % size2
        x1 = i1 % w1
        y1 = i1 / w1
        x2 = i2 % w2
        y2 = i2 / w2
        kernal_radius = int((patch - 1) / 2)

        conv_result = 0
        norm1 = 0
        norm2 = 0
        dy = -kernal_radius
        dx = -kernal_radius
        while dy <= kernal_radius:
            while dx <= kernal_radius:
                xx1 = x1 + dx
                yy1 = y1 + dy
                xx2 = x2 + dx
                yy2 = y2 + dy
                if 0 <= xx1 < w1 and 0 <= yy1 < h1 and 0 <= xx2 < w2 and 0 <= yy2 < h2:
                    _i1 = yy1 * w1 + xx1
                    _i2 = yy2 * w2 + xx2
                    for c in range(0, c1):
                        term1 = input[int(c * size1 + _i1)]
                        term2 = target[int(c * size2 + _i2)]
                        conv_result += term1 * term2
                        norm1 += term1 * term1
                        norm2 += term2 * term2
                dx += stride
            dy += stride
        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)
        conv[i] = conv_result / (norm1 * norm2 + 1e-9)

    match = torch.tensor((), dtype=torch.float32)
    match = match.new_zeros(input.size())

    correspondence = torch.tensor((), dtype=torch.int16)
    correspondence.new_zeros((h1, w1, 2))
    correspondence.resize_(h1 * w1 * 2)

    for id1 in range(0, size1):
        conv_max = -1e20
        for y2 in range(0, h2):
            for x2 in range(0, w2):
                id2 = y2 * w2 + x2
                id = id1 * size2 + id2
                conv_result = conv[id1]

                if conv_result > conv_max:
                    conv_max = conv_result
                    correspondence[id1 * 2 + 0] = x2
                    correspondence[id1 * 2 + 1] = y2

                    for c in range(0, c1):
                        match[c * size1 + id1] = target[c * size2 + id2]

    match.resize_((c1, h1, w1))

    return match, correspondence


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


def blow_up_details(input_image, pos, size, upscale_factor):
    '''
    将图像指定位置的细节放大指定倍数，绘制与图像的左下角并输出到指定路径

    :param input_image: 输入图像
    :param pos: 放大位置的左上角坐标（图像左上角为（0，0）点）
    :param size: 放大区域大小
    :param upscale_factor: 放大倍数
    :return:
    '''
    print("图像大小：{}".format(input_image.size))
    im_box = input_image.crop((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]))
    im_box = im_box.resize((size[0] * upscale_factor, size[1] * upscale_factor))
    h0 = input_image.size[1] - size[1] * upscale_factor
    input_image.paste(im_box, (0, h0))
    im_draw = ImageDraw.Draw(input_image)
    im_draw.line((pos[0], pos[1], pos[0] + size[0], pos[1]), width=1, fill=(255, 0, 0))
    im_draw.line((pos[0], pos[1] + size[1], pos[0] + size[0], pos[1] + size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((pos[0], pos[1], pos[0], pos[1] + size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((pos[0] + size[0], pos[1], pos[0] + size[0], pos[1] + size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((0, h0, size[0] * upscale_factor, h0), width=1, fill=(255, 0, 0))
    im_draw.line((0, input_image.size[1], size[0] * upscale_factor, input_image.size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((0, h0, 0, input_image.size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((size[0] * upscale_factor, h0, size[0] * upscale_factor, input_image.size[1]), width=1,
                 fill=(255, 0, 0))
    return input_image


def image_concat(image_list, image_text_list, output_path):
    '''

    :param image_list:
    :param image_text_list:
    :param output_path:
    :return:
    '''
    target_size = [0, 0]
    x_interval = 20
    y_interval = 50
    target_size[0] += (image_list[0].size[0] + x_interval) * len(image_list) - x_interval
    target_size[1] += image_list[0].size[1] + y_interval
    # 构建画布
    output_image = Image.new('RGB', (target_size[0], target_size[1]), color=(255, 255, 255))
    draw = ImageDraw.Draw(output_image)
    # 设置字体，字号大小
    font = ImageFont.truetype("fonts/arial.ttf", 20)
    x, y = 0, 0
    for i in range(len(image_list)):
        output_image.paste(image_list[i], (x, y))
        draw.text((x, image_list[0].size[1]), image_text_list[i], (0, 0, 0), font=font)
        x += image_list[0].size[0] + x_interval
    output_image.save(output_path)


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

    hr_image = origin_image.resize((image_width, image_height), resample=Image.BICUBIC)
    lr_image = origin_image.resize((image_width // upscale_factor, image_height // upscale_factor),
                                   resample=Image.BICUBIC)
    # lr_image = lr_transform((image_height, image_width), upscale_factor)(origin_image)
    # hr_image = hr_transform((image_height, image_width))(origin_image)

    bicubic = lr_image.resize((image_width, image_height), resample=Image.BICUBIC)
    psnr = calculate_psnr(Variable(ToTensor()(hr_image)).to(device),
                          Variable(ToTensor()(bicubic)).to(device))
    bicubic.save(img_name + "_bicubic_x{}".format(upscale_factor) + suffix)
    # bicubic = transforms.Resize((image_height, image_width))(lr_image)
    # psnr = calculate_psnr(hr_image.to(device), bicubic.to(device))
    # hr_y, _, _ = transforms.ToPILImage()(hr_image).convert('YCbCr').split()
    # bicubic_y, _, _ = transforms.ToPILImage()(bicubic).convert('YCbCr').split()
    hr_y, _, _ = hr_image.convert('YCbCr').split()
    bicubic_y, _, _ = bicubic.convert('YCbCr').split()
    ssim = calculate_ssim(Variable(ToTensor()(hr_y)).to(device),
                          Variable(ToTensor()(bicubic_y)).to(device))
    # transforms.ToPILImage()(bicubic).convert('RGB').save(img_name + "_bicubic_x{}".format(upscale_factor) + suffix)

    # print('bicubic PSNR: {}'.format(psnr))
    # print("bicubic SSIM: {}".format(ssim))

    x = Variable(ToTensor()(lr_image)).to(device).unsqueeze(0)  # 补上batch_size那一维
    y = Variable(ToTensor()(hr_image)).to(device)
    # x = lr_transform((image_height, image_width), upscale_factor)(origin_image).to(device).unsqueeze(0)
    # y = hr_transform((image_height, image_width))(origin_image).to(device)
    # print(x)
    # print(y)
    # x = lr_image.to(device).unsqueeze(0)  # 补上batch_size那一维
    # y = hr_image.to(device)
    # medianpool = SubPixelConvolution.model.MedianPool2d(3)
    with torch.no_grad():
        # out = model(x).clip(0, 1).squeeze()
        out = model(x).clip(0, 1)
        # out = medianpool(out)
        out = out.squeeze()
    # out, y = histogram_match(out, y)
    # out = deprocess((image_height, image_width))(out)
    # y = deprocess((image_height, image_width))(y)
    psnr = calculate_psnr(y, out)
    out_y, _, _ = transforms.ToPILImage()(out).convert('YCbCr').split()
    # print(out_y.size)
    # print(hr_y.size)
    ssim = calculate_ssim(Variable(ToTensor()(hr_y)).to(device),
                          Variable(ToTensor()(out_y)).to(device))
    print('{} PSNR: {}'.format(save_name, psnr))
    print('{} SSIM: {}'.format(save_name, ssim))
    out = tensor_to_image(out)
    out.save(img_name + '_{}_x{}'.format(save_name, upscale_factor) + suffix)
    return
