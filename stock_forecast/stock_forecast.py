#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# @Author: Bosco Lam

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation

data = pd.read_csv('000777.csv', encoding='gbk', parse_dates=[0], index_col=0)
data.sort_index(0, ascending=True, inplace=True)

dayfeature = 150
featurenum = 5 * dayfeature
x = np.zeros((data.shape[0] - dayfeature, featurenum + 1))
y = np.zeros((data.shape[0] - dayfeature))

for i in range(0, data.shape[0] - dayfeature):
    x[i, 0:featurenum] = np.array(data[i:i + dayfeature] \
                                      [[u'收盘价', u'最高价', u'最低价', u'开盘价', u'成交量']]).reshape((1, featurenum))
    x[i, featurenum] = data.ix[i + dayfeature][u'开盘价']

for i in range(0, data.shape[0] - dayfeature):
    if data.ix[i + dayfeature][u'收盘价'] >= data.ix[i + dayfeature][u'开盘价']:
        y[i] = 1
    else:
        y[i] = 0

# 调用svm函数，默认kernel是rbf，其他linear, poly, sigmoid
clf = svm.SVC(kernel='rbf')
result = []
for i in range(5):
    # 将验证集和测试集分成8比2
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
    clf.fit(x_train, y_train)
    # 预测数据和测试集验证数据进行比对
    result.append(np.mean(y_test == clf.predict(x_test)))
print("svm分类器预测准确率:")
print(result)
