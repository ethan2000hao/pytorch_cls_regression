# -*- coding: utf-8 -*-
# @Time : 2021/9/14 14:26
# @Author : jiangwei hao
# @File : create_labels.py
# @Software: PyCharm

import cv2
import numpy as np
import math
import random,string
import shutil
# from __future__ import print_function, division
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

net = ShuffleNetV2()
# net = net.cuda()
net.eval()
# modelpath = sys.argv[1]
modelpath = "model/shuffle2.pt"
print('modelpath:',modelpath)
net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath, map_location=lambda storage, loc: storage).items()})
# print('net:',net)

def randomStr(num):
    # salt = ''.join(random.sample(string.ascii_letters + string.digits, num))
    salt = ''.join(random.sample(string.ascii_lowercase + string.digits, num))
    return salt

def img_check(img_path):

    original_img = cv2.imread(img_path)
    show = cv2.resize(original_img, (128, 128))
    cv2.imwrite('data/re.jpg',show)

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    # img_resize = transform.resize(original_img, (128, 256)) # transform.resize float64 (0~1)
    img_resize = cv2.resize(original_img, (128, 128)) / 255

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
    # print('keypoint:',keypoint)
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
    return cls_nb, keypoint












img = np.ones((255,255,3),np.uint8)*255

# 当鼠标按下时变为True
drawing = False
# 如果mode 为true 绘制矩形。按下'm' 变成绘制曲线。
xy_list = []

# 创建回调函数
def draw_keypoint_label(event, x, y, flags, param):
    global ix, iy, drawing, mode

    # 当按下左键是返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    # 当鼠标左键按下并移动是绘制图形。event 可以查看移动，flag 查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            cv2.circle(img, (x, y), 1, (0, 0, 0), 1)
            xy_list.append(x)
            xy_list.append(y)

    # 当鼠标松开停止绘画。
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:    # 鼠标右键存图
        save_img = np.ones((255, 255, 3), np.uint8) * 255

        for i in range(int(len(xy_list)/2 -1)):
            x1, y1 = xy_list[2*i], xy_list[2*i +1]
            x2, y2 = xy_list[2 * i+2], xy_list[2 * i + 3]
            if math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2) < 600:
                cv2.line(img,(x1,y1),(x2,y2),(255, 0, 0), 1)
                cv2.line(save_img,(x1,y1),(x2,y2),(0, 0, 0),1)
        path = 'data/t.jpg'
        cv2.imwrite(path, save_img)
        cls_nb, keypoint = img_check(path)
        draw_realtime = True
        if draw_realtime:
            cls_name_list = ['circle', 'triangle', 'square', 'hexagon']
            print('图像类别归属：{}'.format(cls_name_list[cls_nb]))

            if cls_nb == 0:
                cv2.putText(img, 'circle', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            elif cls_nb == 1:
                for i in range(3):
                    x, y = int(keypoint[i][0]), int(keypoint[i][1])
                    x = 0 if x < 0 else x
                    x = 128 if x > 128 else x
                    y = 0 if y < 0 else y
                    y = 128 if y > 128 else y
                    cv2.circle(img, (2*x, 2*y), 3, (255, 0, 255), thickness=3)
                    cv2.putText(img, 'triangle', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            elif cls_nb == 2:
                for i in range(4):
                    x, y = int(keypoint[i][0]), int(keypoint[i][1])
                    x = 0 if x < 0 else x
                    x = 128 if x > 128 else x
                    y = 0 if y < 0 else y
                    y = 128 if y > 128 else y
                    cv2.circle(img, (2 * x, 2 * y), 3, (255, 0, 255), thickness=3)
                    cv2.putText(img, 'square', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            elif cls_nb == 3:
                for i in range(6):
                    x, y = int(keypoint[i][0]), int(keypoint[i][1])
                    x = 0 if x < 0 else x
                    x = 128 if x > 128 else x
                    y = 0 if y < 0 else y
                    y = 128 if y > 128 else y
                    cv2.circle(img, (2 * x, 2 * y), 3, (255, 0, 255), thickness=3)
                    cv2.putText(img, 'hexagon', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        print('cls_nb:',cls_nb)
        print('keypoint:',keypoint)
        print('saveing ! \n')
    return x,y


cv2.namedWindow('image')
# cv2.setMouseCallback('image', draw_circle)
while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF

    cv2.setMouseCallback('image', draw_keypoint_label)
    if k == ord('c'):
        print('clear')
        xy_list = []
        img = np.ones((255, 255, 3), np.uint8) * 255
        cv2.setMouseCallback('image', draw_keypoint_label)

    elif k == ord('s'):  # 存图
        print('saved')
        img_name = randomStr(4) + '_' + randomStr(4) + '_' + '.jpg'
        new_path = 'E:/data/Sketch/enhance_drow/imgs/' + img_name
        shutil.copy('./data/t.jpg',new_path)
        xy_list = []
        img = np.ones((255, 255, 3), np.uint8) * 255
        cv2.setMouseCallback('image', draw_keypoint_label)
    elif k == ord('q'):
        break
