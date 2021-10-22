# -*- coding: utf-8 -*-
# @Time : 2021/9/1 16:59 
# @Author : jiangwei hao 
# @File : utils.py 
# @Software: PyCharm

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