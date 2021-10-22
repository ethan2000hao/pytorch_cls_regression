# -*- coding: utf-8 -*-
# @Time : 2021/8/31 18:10 
# @Author : jiangwei hao 
# @File : Resnet.py 
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class VGGRegNet(nn.Module):
    def __init__(self):
        super(VGGRegNet, self).__init__()
        self.conv1_0 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1_0 = nn.BatchNorm2d(16)
        self.conv1_1 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv2_0 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2_0 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv3_0 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3_0 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv4_0 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4_0 = nn.BatchNorm2d(128)
        self.conv4_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.conv5_0 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5_0 = nn.BatchNorm2d(256)
        self.conv5_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv6_0 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6_0 = nn.BatchNorm2d(512)
        self.conv6_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(512)
        self.loc = nn.Conv2d(512, 10, 1, stride=2)

        self.numconv = nn.Conv2d(256, 256, 1, padding=1)
        self.bn_num = nn.BatchNorm2d(256)


    def forward(self, x):  # 3x128x256
        x = F.relu(self.bn1_0(self.conv1_0(x)))
        x = F.max_pool2d(F.relu(self.bn1_1(self.conv1_1(x))), 2)  # 16x64x128
        x = F.relu(self.bn2_0(self.conv2_0(x)))
        # print('x1:', x.shape)
        x = F.max_pool2d(F.relu(self.bn2_1(self.conv2_1(x))), 2)  # 32x32x64
        x = F.relu(self.bn3_0(self.conv3_0(x)))
        # print('x2:', x.shape)
        x = F.max_pool2d(F.relu(self.bn3_1(self.conv3_1(x))), 2)  # 64x16x32
        x = F.relu(self.bn4_0(self.conv4_0(x)))
        # print('x3:', x.shape)
        x = F.max_pool2d(F.relu(self.bn4_1(self.conv4_1(x))), 2)  # 128x8x16
        x = F.relu(self.bn5_0(self.conv5_0(x)))  # 256x8x16
        # print('x5:', x.shape)
        x = F.max_pool2d(F.relu(self.bn5_1(self.conv5_1(x))), 2)  # 256x4x8
        x = F.relu(self.bn6_0(self.conv6_0(x)))  # 512x4x8
        # print('x6:', x.shape)
        x = F.max_pool2d(F.relu(self.bn6_1(self.conv6_1(x))), 2)  # 512x2x4
        # print('x7:', x.shape)
        x = self.loc(x)  # 1*2*16

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

def test():
    net = VGGRegNet()
    # print(net)


    # x = torch.randn(1,3,32,32)
    # x = torch.randn(1, 3, 64, 48)
    x = torch.randn(1, 3, 128, 128)
    # x, y1, y2, y3, y4, y5, y6, y7, y8 = net(x)
    x = net(x)
    print('x:',x.shape)
    print(len(x))
    # print(y.size())


if __name__ == '__main__':
    test()