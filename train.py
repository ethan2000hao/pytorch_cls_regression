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
from dataset import KeyPointDataset, Rescale, ToTensor
from torch.utils.data import Dataset, DataLoader


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

train_path = "/workspace1/haojiangwei/data/PlateKeyPoint/trainmul.txt"
# train_path = "data/trainmul.txt"


kptrain = KeyPointDataset(label_file=train_path, root_dir="/workspace1/haojiangwei/data/PlateKeyPoint/train/",
                              transform=transforms.Compose([Rescale((128, 256)), ToTensor()]))
kpval = KeyPointDataset(label_file="/workspace1/haojiangwei/data/PlateKeyPoint/valmul.txt",root_dir="/workspace1/haojiangwei/data/PlateKeyPoint/val",transform=transforms.Compose([Rescale((128,256)),ToTensor()]))

# DataLoader数据加载
batch_size = 1
trainloader = DataLoader(kptrain, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(kpval, batch_size=batch_size, shuffle=False, num_workers=4)

use_gpu = torch.cuda.is_available()
net_regression = VGGRegNet()
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
initial_lr = 0.001
optimizer = optim.AdamW(net_regression.parameters(), lr=initial_lr)
# optimizer_ft = optim.SGD(net_regression.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)




def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer,iteration):
    lr_ini = 0.000001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_ini+(initial_lr - lr_ini)*iteration/100

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


lr_decay = [2500, 4600, 5200]
step_index = 0
tlall = []
tllall = []
tplall = []
vlall = []
vllall = []
vplall = []
for iteration in range(6000):
    running_loss = 0
    running_loc_loss = 0
    val_loss = 0
    val_loc_loss = 0

    start = time.time()
    batch_iterator = iter(trainloader)
    (images, labels), plate = next(batch_iterator)
    print('labels:',labels[0][1])
    print('-----------------')
    images = images.to(device)
    labels = labels.to(device)
    val_iter = iter(valloader)
    (valimg, vallabels), vplate = next(val_iter)
    print('vallabels:', vallabels[0][1])
    valimg = valimg.to(device)
    vallabels = vallabels.to(device)

    optimizer.zero_grad()
    # outputs, p1, p2, p3, p4, p5, p6, p7, p8 = net(images)
    outputs = net_regression(images)
    #     print(outputs.size())
    #     print(labels.size())
    outputs = outputs.squeeze(2)
    #     print(outputs.size())
    loss0 = criterion(outputs, labels)
    ############# 识别损失函数#############

    locloss = loss0
    # plateloss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
    # loss = locloss + plateloss
    loss = locloss

    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    running_loc_loss += locloss.item()

    val_out = net_regression(valimg)
    val_out = val_out.squeeze(2)
    vallocloss = criterion(val_out, vallabels)



    # vploss = vloss1 + vloss2 + vloss3 + vloss4 + vloss5 + vloss6 + vloss7 + vloss8
    # valloss = vallocloss + vploss
    valloss = vallocloss
    val_loss += valloss.item()
    val_loc_loss += vallocloss.item()


    if iteration <= 100:
        warmup_learning_rate(optimizer, iteration)
    if iteration in lr_decay:
        step_index += 1
        adjust_learning_rate(optimizer, 0.2, step_index)
    for param in optimizer.param_groups:
        if 'lr' in param.keys():
            if iteration % 10 == 9:
                tlall.append(running_loss / 10)
                vlall.append(val_loss / 10)
                tllall.append(running_loc_loss / 10)
                tplall.append(running_plate_loss / 10)
                vllall.append(val_loc_loss / 10)
                vplall.append(val_plate_loss / 10)
                print(
                    "Iter %d  || LR %f || TrainLoss %.4f || TrainLocLoss %.4f || TrainClsLoss %.4f || ValLoss %.4f || ValLocLoss %.4f || ValClsLoss %.4f || Time: %f" % (
                    iteration + 1, param['lr'], running_loss / 10, running_loc_loss / 10, running_plate_loss / 10,
                    val_loss / 10, val_loc_loss / 10, val_plate_loss / 10, time.time() - start))
    running_loss = 0
    running_loc_loss = 0
    running_plate_loss = 0
    val_loss = 0
    val_loc_loss = 0
    val_plate_loss = 0

print("Finish Training...")

torch.save(net_regression.state_dict(), 'model/draw_vgg.pt')
torch.save(net_regression, "model/draw_vgg1.pt")