# -*- coding: utf-8 -*-
# @Time : 2021/9/2 15:55 
# @Author : jiangwei hao 
# @File : infrence.py 
# @Software: PyCharm

from __future__ import print_function, division
from net.Resnet import VGGRegNet
import os
from skimage import io,transform
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

data_transforms = transforms.Compose([
            transforms.Resize((60, 60)),
            transforms.ToTensor()])


net = VGGRegNet()
# net = net.cuda()
net.eval()
# modelpath = sys.argv[1]
# modelpath = "model/draw_vgg.pt"
modelpath = "model/draw.pt"
print('modelpath:',modelpath)
net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath, map_location=lambda storage, loc: storage).items()})
# print('net:',net)

# img1 = io.imread('data/0001.jpg')
original_img = cv2.imread('data/0004.jpg')

show = cv2.resize(original_img, (256, 128))
cv2.imwrite('data/re.jpg',show)

original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# img_resize = transform.resize(original_img, (128, 256)) # transform.resize float64 (0~1)
img_resize = cv2.resize(original_img, (256, 128)) / 255



print(img_resize.shape)  # shape HWC
img = img_resize.transpose((2, 0, 1)) # (3, 128, 256) shape CHW
print(img.shape)
img = np.array(img)
img = torch.from_numpy(img)

img = img.unsqueeze(0)
img = img.type(torch.FloatTensor)
# img = img.cuda()
predict = net(img)

keyp = predict

keyp = keyp.squeeze(2)
keyp = keyp.squeeze(0)
pkp = keyp.cpu().detach().numpy()
pkp = pkp.reshape(-1,2)
# plt.imshow(img1)
plt.scatter(pkp[:, 0], pkp[:, 1], s=30, marker='.', c='r',linewidths = 5)
# plt.imsave('a.jpg',img)
print(pkp)
print(len(pkp[0]))
print(pkp[0])
for i in range(len(pkp)):
    x, y = int(pkp[i][0]), int(pkp[i][1])
    cv2.circle(show,(x,y),3,(255,255,255),thickness=3)

cv2.imwrite('data/test1.jpg',show)
print('finish')