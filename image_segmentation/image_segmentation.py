#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# @Author: Bosco Lam

import numpy as np
import PIL.Image as Image
from sklearn.cluster import KMeans


def load_data(file_path):
    f = open(file_path, 'rb')
    data = []
    img = Image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return np.mat(data), m, n


imgData, row, col = load_data('real_madrid.jpg')
# 获得每个像素所属类别
label = KMeans(n_clusters=4).fit_predict(imgData)
label = label.reshape([row, col])
# 创建新的灰度图保存聚类后的结果
pic_new = Image.new("L", (row, col))
# 根据所属类别向图片中添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
pic_new.save("result-rm-4.jpg", "JPEG")
