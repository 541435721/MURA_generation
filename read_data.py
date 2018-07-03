#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_data.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  : 2018/6/28 16:49
# @Desc  : 

import cv2
import os
import time
import numpy as np

baseURL = '../../Medical_Image_Data/MURA-v1.1/MURA-v1.1/'


def read_X_Ray_image(url: str) -> list:
    img_lists = []
    sub_dir_names = list(filter(lambda x: os.path.isdir(baseURL + x), os.listdir(url)))
    big = [100, 100]
    for sub_dir_name in sub_dir_names:
        for fpathe, dirs, fs in os.walk(baseURL + sub_dir_name):
            for f in fs:
                img_url = os.path.join(fpathe, f)
                img = cv2.imread(img_url, 0)
                img = cv2.resize(img, (img.shape[0] // 8, img.shape[1] // 8))
                new_img = np.zeros([64, 64])
                if img.shape[0] < 50 or img.shape[1] < 50:
                    break
                new_img[0:img.shape[0], 0:img.shape[1]] = img
                img_lists.append(new_img)
                # big = [img.shape[0] if img.shape[0] < big[0] else big[0],
                #        img.shape[1] if img.shape[1] < big[1] else big[1]]
                # print(big)
    return img_lists


if __name__ == '__main__':
    img_list = read_X_Ray_image(baseURL)

    np.save('x_ray_resized_equal_good_shape.npy', img_list)
