#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# @Author: Bosco Lam

import numpy as np
from sklearn.cluster import KMeans


def load_data(file_path):
    fr = open(file_path, 'r+')
    lines = fr.readlines()
    ret_data = []
    ret_city_name = []
    for line in lines:
        items = line.strip().split(",")
        ret_city_name.append(items[0])
        ret_data.append([float(items[i]) for i in range(1, len(items))])
    return ret_data, ret_city_name


if __name__ == '__main__':
    data, cityName = load_data('city.txt')
    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_, axis=1)
    CityCluster = [[], [], [], []]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])