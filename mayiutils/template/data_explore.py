#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore.py
@time: 2019-09-02 11:54
"""
import numpy as np
import sys
sys.path.append('/Users/luoyonggui/PycharmProjects/mayiutils_n1/mayiutils/data_prepare')
from date_prepare import describe
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全
pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行

df = pd.DataFrame()

df.head()

df_info = describe(df)

# 查看每个特征的分布
dftrain.age.hist(bins=20);
dftrain.sex.value_counts().plot(kind='barh');
(dftrain['class']
  .value_counts()
  .plot(kind='barh'));

# 查看两个特征间的关系
sns.pairplot()



