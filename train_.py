# -*- coding: utf-8 -*-
# @Time : 2021/9/1 10:30 
# @Author : jiangwei hao 
# @File : train.py 
# @Software: PyCharm


from __future__ import print_function, division

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from net.Resnet import VGGRegNet
from net.ShuffleV2 import ShuffleNetV2
from dataset import KeyPointDataset, Rescale, ToTensor
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


import time
time_obj = time.localtime()
pic_name = str(time_obj.tm_year) + '_' + str(time_obj.tm_mon) + str(time_obj.tm_mday) + '_' + str(time_obj.tm_hour) + '_' + str(time_obj.tm_min) + '.jpg'
print(pic_name)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

initial_lr = 0.01
batch_size = 320
num_epochs = 190
height = 128
width = 128
# height = 255
# width = 255


train_txt_path = "/workspace1/data/keypoint/0915_keypoint_4/train_cls_4.txt"
train_images = "/workspace1/data/keypoint/0915_keypoint_4/train/"
val_txt_path = "/workspace1/data/keypoint/0915_keypoint_4/val_cls_4.txt"
val_images = "/workspace1/data/keypoint/0915_keypoint_4/val/"
# train_path = "data/trainmul.txt"
kptrain = KeyPointDataset(label_file=train_txt_path, root_dir=train_images,
                              transform=transforms.Compose([Rescale((height, width)), ToTensor()]))
kpval = KeyPointDataset(label_file=val_txt_path, root_dir=val_images, transform=transforms.Compose([Rescale((height, width)), ToTensor()]))

# DataLoader数据加载

trainloader = DataLoader(kptrain, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(kpval, batch_size=batch_size, shuffle=True, num_workers=4)

use_gpu = torch.cuda.is_available()
# net_regression = VGGRegNet()
net_regression = ShuffleNetV2()
print(net_regression)

CUDA: 3
# device = torch.device("cuda:0,1")  # 指定模型训练所在 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定模型训练所在 GPU
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # modelclc = nn.DataParallel(modelclc, device_ids=[0, 2])   # 多GPU
    net_regression = nn.DataParallel(net_regression, device_ids=[0])  # 单GPU

net_regression.cuda(device)

criterion = nn.SmoothL1Loss()   # regression loss
criterion2 = nn.CrossEntropyLoss()  # classification loss

optimizer = optim.AdamW(net_regression.parameters(), lr=initial_lr)
# optimizer = optim.SGD(net_regression.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)


step_index = 0
tlall = []
tllall = []
tplall = []
vlall = []
vllall = []
vplall = []

# print('trainloader nums {}/{}'.format(len(trainloader[0][0])))

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    lrarte = optimizer.state_dict()['param_groups'][0]['lr']
    running_loss = 0
    running_loc_loss = 0
    running_cls_loss = 0
    testing_val_loss = 0
    testing_val_loc_loss = 0
    testing_val_cls_loss = 0
    start = time.time()

    for i, (sample, cls) in enumerate(trainloader):
        images, labels = sample[0], sample[1]
        images = images.to(device)
        labels = labels.to(device)
        cls = cls.to(device)
        cls = cls.squeeze() # CrossEntropyLoss  输入的正确 label (target)不能是 one-hot([1,0,2]) 格式。所以只需要输入数字(1,0,2)  就行


        labels = labels.view(len(labels), 12)
        outputs = net_regression(images)
        # print(outputs.size())
        # outputs = outputs.squeeze(2)
        outputs = outputs.squeeze()
        outputs_keypoints = outputs[:, :12]
        # print('outputs_keypoints.size:',outputs_keypoints.size())
        outputs_cls = outputs[:, 12:]

        for j in range(len(cls)):
            if cls[j] == 0:
                outputs_keypoints[j][0], outputs_keypoints[j][1] = 0, 0
                outputs_keypoints[j][2], outputs_keypoints[j][3] = 0, 0
                outputs_keypoints[j][4], outputs_keypoints[j][4] = 0, 0
                outputs_keypoints[j][6], outputs_keypoints[j][7] = 0, 0
                outputs_keypoints[j][8], outputs_keypoints[j][9] = 0, 0
                outputs_keypoints[j][10], outputs_keypoints[j][11] = 0, 0
            elif cls[j] == 1:
                outputs_keypoints[j][6], outputs_keypoints[j][7] = 0, 0
                outputs_keypoints[j][8], outputs_keypoints[j][9] = 0, 0
                outputs_keypoints[j][10], outputs_keypoints[j][11] = 0, 0
            elif cls[j] == 2:
                outputs_keypoints[j][8], outputs_keypoints[j][9] = 0, 0
                outputs_keypoints[j][10], outputs_keypoints[j][11] = 0, 0
        # print('outputs_keypoints:',outputs_keypoints)

        loss_keypoints = criterion(outputs_keypoints, labels)
        loss_cls = criterion2(outputs_cls, cls.long())

        ############# 识别损失函数#############

        loss = loss_keypoints + 0.1*loss_cls

        optimizer.zero_grad()            ## 清空过往梯度
        loss.backward()                  ## 反向传播，计算当前梯度；
        optimizer.step()                 ## 根据梯度更新网络参数
        running_loss += loss.item()
        running_loc_loss += loss_keypoints.item()
        running_cls_loss += loss_cls.item()

    for i, (sample, val_cls) in enumerate(valloader):
        valimg, vallabels = sample[0], sample[1]

        valimg = valimg.to(device)
        vallabels = vallabels.to(device)
        val_cls = val_cls.to(device)

        vallabels = vallabels.view(len(vallabels), 12)
        val_cls = val_cls.squeeze()

        val_out = net_regression(valimg)
        val_out = val_out.squeeze()
        val_outputs_keypoints = val_out[:, :12]
        # print('outputs_keypoints.size:',outputs_keypoints.size())
        val_outputs_cls = val_out[:, 12:]
        for j in range(len(val_cls)):
            if val_cls[j] == 0:
                val_outputs_keypoints[j][0], val_outputs_keypoints[j][1] = 0, 0
                val_outputs_keypoints[j][2], val_outputs_keypoints[j][3] = 0, 0
                val_outputs_keypoints[j][4], val_outputs_keypoints[j][4] = 0, 0
                val_outputs_keypoints[j][6], val_outputs_keypoints[j][7] = 0, 0
                val_outputs_keypoints[j][8], val_outputs_keypoints[j][9] = 0, 0
                val_outputs_keypoints[j][10], val_outputs_keypoints[j][11] = 0, 0
            elif val_cls[j] == 1:
                val_outputs_keypoints[j][6], val_outputs_keypoints[j][7] = 0, 0
                val_outputs_keypoints[j][8], val_outputs_keypoints[j][9] = 0, 0
                val_outputs_keypoints[j][10], val_outputs_keypoints[j][11] = 0, 0
            elif val_cls[j] == 2:
                val_outputs_keypoints[j][8], val_outputs_keypoints[j][9] = 0, 0
                val_outputs_keypoints[j][10], val_outputs_keypoints[j][11] = 0, 0

        val_loc_loss = criterion(val_outputs_keypoints, vallabels)
        val_cls_loss = criterion2(val_outputs_cls, val_cls.long())

        val_loss = val_loc_loss + val_cls_loss
        # valloss = val_cls_loss
        # print('valloss:',valloss)


        testing_val_loss += val_loss.item()
        testing_val_loc_loss += val_loc_loss.item()
        testing_val_cls_loss += val_cls_loss.item()



    tlall.append(running_loss / 1)
    tllall.append(running_loc_loss / 1)
    tplall.append(running_cls_loss / 1)

    vlall.append(testing_val_loss / 1)
    vllall.append(testing_val_loc_loss / 1)
    vplall.append(testing_val_cls_loss / 1)
    print(
        "Iter %d  || LR %f || TrainLoss %.4f || Train-Loc-Loss %.4f || Train-Cls-Loss %.4f || ValLoss %.4f || Val-Loc-Loss %.4f || Val-Cls-Loss %.4f || Time: %f" % (
        epoch + 1, lrarte, tlall[-1], tllall[-1], tplall[-1], vlall[-1], vllall[-1], vplall[-1], time.time() - start))


    exp_lr_scheduler.step()

x1 = range(0, num_epochs)
x2 = range(0, num_epochs)
y1 = tlall
y2 = vlall
# plt.subplot(2,1,1)
plt.plot(x1, y1, 'r-')
plt.plot(x2, y2, 'b-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.savefig('model/'+ pic_name)


print("Finish Training...")

torch.save(net_regression.state_dict(), 'model/shuffle2_sm.pt')
