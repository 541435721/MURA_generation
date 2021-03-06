#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : normal_GAN_2_modified_5.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  : 2018/7/2 22:57
# @Desc  : add more fc layers to D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import tensorboardX
import torchvision.utils as vutils

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class upsample(nn.Module):
    def __init__(self, in_channel, kernel, stride, padding):
        super(upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel, stride, padding)
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


class Generator(nn.Module):
    """"""

    def __init__(self, in_channel, out_channel, kernel, stride, padding=0):
        """Constructor for Generator"""
        super(Generator, self).__init__()
        self.linear = nn.Linear(in_channel, in_channel * 20)
        self.deconv_1 = upsample(in_channel * 20, kernel, stride, padding)
        self.deconv_2 = upsample(in_channel * 20 // 2, kernel, stride, padding)
        self.deconv_3 = upsample(in_channel * 20 // 4, kernel, stride, padding)
        self.deconv_4 = upsample(in_channel * 20 // 8, kernel, stride, padding)
        self.deconv_5 = upsample(in_channel * 20 // 16, kernel, stride, padding)
        self.deconv_6 = upsample(in_channel * 20 // 32, kernel, stride, padding)
        self.conv = nn.Conv2d(in_channel * 20 // 64, out_channel, 1, 1)

    def forward(self, input):
        out = input
        out = F.leaky_relu(self.linear(out), 0.2)
        out = out.view(-1, out.size(1), 1, 1)
        out = self.deconv_1(out)
        out = self.deconv_2(out)
        out = self.deconv_3(out)
        out = self.deconv_4(out)
        out = self.deconv_5(out)
        out = self.deconv_6(out)
        out = F.tanh(self.conv(out))

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


class Discriminator(nn.Module):
    """"""

    def __init__(self, in_channel, kernel, stride, padding=0):
        """Constructor for Distriminator"""
        super(Discriminator, self).__init__()
        self.conv_1 = conv_layer(in_channel, 32, kernel, stride, padding)
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
        out = F.leaky_relu(self.bn_1(F.dropout(self.linear_1(out))))  # add dropout and bn
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
    G = Generator(200, 1, 2, 2)
    D = Discriminator(1, 3, 1)

    G_Optim = Adam(G.parameters(), 0.0002, betas=(0.5, 0.999))
    D_Optim = Adam(D.parameters(), 0.0002, betas=(0.5, 0.999))

    G_Optim_scheduler = lr_scheduler.StepLR(G_Optim, step_size=200, gamma=0.95)
    D_Optim_scheduler = lr_scheduler.StepLR(D_Optim, step_size=200, gamma=0.95)

    loss_F = nn.BCELoss()

    import numpy as np

    ##########################
    batch_size = 128
    times_for_D = 3
    count = 0
    ##########################

    data = np.load('x_ray_resized_equal_good_shape.npy')[:, np.newaxis, ...] / 255 * 2 - 1
    summary = tensorboardX.SummaryWriter('./normal_GAN_log_2_modified_5')
    for epoch in range(5000):
        out_img = None
        G_loss = 0
        D_loss = 0
        real_sample = None
        for index in range(0, data.shape[0], batch_size):
            count += 1
            real_sample = torch.from_numpy(data[index:index + batch_size, ...]).float()
            z = torch.randn(real_sample.size(0), 200)  # noise

            D_fake_label = torch.Tensor(real_sample.size(0), 1).uniform_(0, 0.3)  # torch.zeros(z.size(0), 1) + 0.3
            D_real_label = torch.Tensor(real_sample.size(0), 1).uniform_(1 - 0.3, 1)  # torch.ones(z.size(0), 1) - 0.3
            G_fake_label = torch.Tensor(real_sample.size(0), 1).uniform_(1 - 0.3, 1)  # torch.ones(z.size(0), 1) - 0.3

            if torch.cuda.is_available():
                z = z.cuda()
                real_sample = real_sample.cuda()

                D_fake_label = D_fake_label.cuda()
                D_real_label = D_real_label.cuda()
                G_fake_label = G_fake_label.cuda()

                G = G.cuda()
                D = D.cuda()

            for i in range(times_for_D):
                fake_sample = G(z)
                fake_pre = D(fake_sample)
                real_pre = D(real_sample)

                D_Optim.zero_grad()
                D_loss = (torch.mean(loss_F(fake_pre, D_fake_label)) + torch.mean(loss_F(real_pre, D_real_label))) * 0.5
                D_loss.backward()
                D_Optim.step()

            fake_sample = G(z)
            fake_pre = D(fake_sample)
            out_img = fake_sample
            G_Optim.zero_grad()
            G_loss = torch.mean(loss_F(fake_pre, G_fake_label))
            G_loss.backward()
            G_Optim.step()

            summary.add_scalar('gen_loss', G_loss.cpu().data.numpy(), count)
            summary.add_scalar('dis_loss', D_loss.cpu().data.numpy(), count)
            print("Iteration:{0},g_loss:{1},d_loss:{2}".format(count, G_loss.cpu().data, D_loss.cpu().data))

        G_Optim_scheduler.step()
        D_Optim_scheduler.step()

        fake_img = vutils.make_grid((out_img + 1) / 2.0, normalize=True, scale_each=True)
        real_img = vutils.make_grid((real_sample + 1) / 2.0, normalize=True, scale_each=True)
        summary.add_image('fake_image', fake_img.cpu(), epoch)
        summary.add_image('real_image', real_img.cpu(), epoch)

        torch.save(G.state_dict(), 'params_G_2_modified_5.pkl')
        torch.save(D.state_dict(), 'params_D_2_modified_5.pkl')
    summary.close()
