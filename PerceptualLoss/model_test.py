# -*- coding: utf-8 -*-
'''
测试模型效果
'''
# @Time    : 2021/8/5 15:07
# @Author  : LINYANZHEN
# @File    : model_test.py
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from torch.autograd import Variable
import utils
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    upscale_factor = 3
    test_image_path = "../img_test/test.jpg"
    save_path = "test"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load("../checkpoint/sub_pixel_convolution.pth").to(device)
    model.eval()

    origin_image = Image.open(test_image_path).convert('RGB')
    image_width = (origin_image.width // upscale_factor) * upscale_factor
    image_height = (origin_image.height // upscale_factor) * upscale_factor
    hr_image = origin_image.resize((image_width, image_height), resample=Image.BICUBIC)
    lr_image = origin_image.resize((image_width // upscale_factor, image_height // upscale_factor),
                                   resample=Image.BICUBIC)

    bicubic = lr_image.resize((image_width, image_height), resample=Image.BICUBIC)
    psnr = utils.calaculate_psnr(Variable(ToTensor()(origin_image)).to(device),
                                 Variable(ToTensor()(bicubic)).to(device))

    print('bicubic PSNR: {}'.format(psnr))
    bicubic.save(test_image_path.replace(".jpg", "_bicubic_x{}.jpg".format(upscale_factor)))

    x = Variable(ToTensor()(lr_image)).to(device).unsqueeze(0)
    y = Variable(ToTensor()(origin_image)).to(device)
    with torch.no_grad():
        out = model(x).clip(0, 1).squeeze()
    psnr = utils.calaculate_psnr(y, out)
    print('espcn PSNR: {}'.format(psnr))
    out = utils.tensor_to_image(out)
    out.save(test_image_path.replace('.jpg', '_self_x{}.jpg'.format(upscale_factor)))
