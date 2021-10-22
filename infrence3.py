# -*- coding: utf-8 -*-
# @Time : 2021/9/2 15:55 
# @Author : jiangwei hao 
# @File : infrence.py 
# @Software: PyCharm

from __future__ import print_function, division
from net.Resnet import VGGRegNet
from net.ShuffleV2 import ShuffleNetV2
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

data_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

# net = VGGRegNet()
net = ShuffleNetV2()
# net = net.cuda()
net.eval()
# modelpath = sys.argv[1]
modelpath = "model/shuffle2_sm.pt"
print('modelpath:',modelpath)
original_img = cv2.imread('data/test/c1.jpg')
# original_img = cv2.imread('data/test/c5.jpg')


net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath, map_location=lambda storage, loc: storage).items()})
# print('net:',net)
show = cv2.resize(original_img, (128, 128))
cv2.imwrite('data/re.jpg',show)

original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
# img_resize = transform.resize(original_img, (128, 256)) # transform.resize float64 (0~1)
img_resize = cv2.resize(original_img, (128, 128)) / 255
print(img_resize[0][0])


# print(img_resize.shape)  # shape HWC
img = img_resize.transpose((2, 0, 1)) # (3, 128, 256) shape CHW
# print(img.shape)
img = np.array(img)
img = torch.from_numpy(img)

img = img.unsqueeze(0)
img = img.type(torch.FloatTensor)
# img = img.cuda()
predict = net(img)

keyp = predict
print('keyp:',keyp.shape)
keyp = keyp.squeeze()
pkp = keyp.cpu().detach().numpy()


keypoint = pkp[:12].reshape(-1,2)
print('keypoint:',keypoint)
cls_name_list = ['circle', 'triangle', 'square', 'hexagon']
cls_list = pkp[12:].tolist()
print('cls_list:',cls_list)
cls_nb = cls_list.index(max(cls_list))
print('图像类别归属：{}'.format(cls_name_list[cls_nb]))

if cls_nb == 0:
    cv2.putText(show, 'circle', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

elif cls_nb == 1:
    for i in range(3):
        x, y = int(keypoint[i][0]), int(keypoint[i][1])
        print(x,y)
        x = 0 if x < 0 else x
        x = 128 if x > 128 else x
        y = 0 if y < 0 else y
        y = 128 if y > 128 else y
        cv2.circle(show,(x,y),3,(255,0,255),thickness=3)
        cv2.putText(show, 'triangle', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

elif cls_nb == 2:
    for i in range(4):
        x, y = int(keypoint[i][0]), int(keypoint[i][1])
        x = 0 if x < 0 else x
        x = 128 if x > 128 else x
        y = 0 if y < 0 else y
        y = 128 if y > 128 else y
        cv2.circle(show, (x, y), 3, (255, 0, 255), thickness=3)
        cv2.putText(show, 'square', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

elif cls_nb == 3:
    for i in range(6):
        x, y = int(keypoint[i][0]), int(keypoint[i][1])
        x = 0 if x < 0 else x
        x = 128 if x > 128 else x
        y = 0 if y < 0 else y
        y = 128 if y > 128 else y
        cv2.circle(show, (x, y), 3, (255, 0, 255), thickness=3)
        cv2.putText(show, 'hexagon', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

cv2.imwrite('data/result.jpg',show)
print('finish')




