# -*- coding: utf-8 -*-
'''
直方图匹配
'''
# @Time    : 2021/11/18 21:33
# @Author  : LINYANZHEN
# @File    : histogram_match.py
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision import transforms


def histogram_match(input_tensor, target_tensor, bins=1024):
    input_size = input_tensor.shape[0] * input_tensor.shape[1]
    target_size = target_tensor.shape[0] * target_tensor.shape[1]
    print(input_size)
    print(target_size)
    input_histc = torch.histc(input_tensor, bins, 0, 1) / input_size
    target_histc = torch.histc(target_tensor, bins, 0, 1) / target_size
    # sk代表输入图像的灰度级概率密度函数积分
    sk_list = []
    for k in range(bins):
        sk = 0
        for j in range(k + 1):
            sk += input_histc[j]
        sk *= (bins - 1)
        sk_list.append(sk.item())
    print(sk_list)
    # print(sk_list)
    # G代表目标图像的灰度级概率密度函数积分
    g_list = []
    for i in range(bins):
        g = 0
        for j in range(i + 1):
            g += target_histc[j]
        g *= (bins - 1)
        g_list.append(g.item())
    # print(g_list)
    # sk_z代表输入图像的像素值与目标图像的像素值之间的对应关系
    sk_z = []
    for sk in sk_list:
        z = [abs(i - sk) for i in g_list]
        sk_z.append(z.index(min(z)))
    # 对输入图像进行直方图均衡
    # 对输入图像每一个像素进行映射
    # print(input_img)
    input_tensor.apply_(lambda x: sk_list[round(x * (bins - 1))])
    # print(input_img)
    input_tensor.apply_(lambda x: sk_z[round(x)] / (bins - 1))
    return input_tensor


if __name__ == '__main__':
    # 图片导入
    input_img_path = "img_test/test.png"
    target_img_path = "img_test/test.png"
    input_img = Variable(ToTensor()(Image.open(input_img_path).convert("RGB")))
    target_img = Variable(ToTensor()(Image.open(target_img_path).convert("RGB")))
    input_img_size = input_img.shape[0] * input_img.shape[1] * input_img.shape[2]
    target_img_size = target_img.shape[0] * target_img.shape[1] * target_img.shape[2]
    output_img = torch.stack(
        [histogram_match(input_img[i, :, :], target_img[i, :, :]) for i in range(input_img.shape[0])], dim=0)
    output_img = transforms.ToPILImage()((input_img).clamp(0, 1))
    output_img.save("img_test/test_histogram_match.png")
