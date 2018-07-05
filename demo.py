#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  : 2018/7/3 10:22
# @Desc  :

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import tensorboardX
import torchvision.utils as vutils

import os


class upsample(nn.Module):
    def __init__(self, in_channel, kernel, stride, padding):
        super(upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channel, in_channel // 2, kernel, stride, padding)
        self.dropout = nn.Dropout2d()
        self.bn = nn.BatchNorm2d(in_channel // 2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        out = input
        out = self.conv(out)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class conv_layer(nn.Module):
    """"""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        """Constructor for conv_layer"""
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding)
        self.dropout = nn.Dropout2d()
        self.bn = nn.BatchNorm2d(out_channel)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        out = input
        out = self.conv(out)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.lrelu(out)
        return out


class Discriminator(nn.Module):
    """"""

    def __init__(self, in_channel, kernel, stride, padding=0):
        """Constructor for Distriminator"""
        super(Discriminator, self).__init__()
        self.conv_1 = conv_layer(in_channel, 32, kernel + 2, stride, padding)
        self.conv_2 = conv_layer(32, 64, kernel, 2, padding)
        self.conv_3 = conv_layer(64, 128, kernel, stride, padding)
        self.conv_4 = conv_layer(128, 256, kernel, 2, padding)
        self.linear_1 = nn.Linear(43264, 1024)
        self.bn_1 = nn.BatchNorm1d(1024)
        self.linear_2 = nn.Linear(1024, 1)

    def forward(self, input):
        out = input
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = out.view(out.size(0), -1)
        # out = F.max_pool2d(out, out.size()[2:])
        out = F.leaky_relu(self.bn_1(self.linear_1(out)))
        out = F.sigmoid(self.linear_2(out))
        return out

    def _init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                w.weight.data.normal_(0, 0.02)
                w.bias.data.zero_()
            if isinstance(w, nn.Linear):
                w.weight.data.normal_(0, 0.02)
                w.bias.data.zero_()
            if isinstance(w, nn.ConvTranspose2d):
                w.weight.data.normal_(0, 0.02)
                w.bias.data.zero_()


if __name__ == '__main__':
    img = torch.zeros(2, 1, 64, 64)
    D = Discriminator(1, 3, 1)
    y = D(img)
    print(y.size())
    pass
