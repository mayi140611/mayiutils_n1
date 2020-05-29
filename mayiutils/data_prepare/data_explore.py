#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore.py.py
@time: 2019-06-26 10:09
"""
import numpy as np
import pandas as pd


class DataExplore:


    @classmethod
    def normalize(cls, X, norm='01', axis=0, paramdict=dict()):
        """
        这个需要根据example改
        归一化
        :param X: ndarray
        :param norm:'l1', 'l2', or 'max', '01', 'normal' optional ('l2' by default)
            The norm to use to normalize each non zero sample (or each non-zero
            feature if axis is 0).
            '01': 处理后，最小值为0，最大值为1
            'normal'： 处理后，均值为0，标准差为1。注意：此时axis无效，只能处理column！
        :param axis:0 or 1, optional (0 by default)
            axis used to normalize the data along. If 1, independently normalize
            each sample, otherwise (if 0) normalize each feature.
        :return:
        """
        if norm == '01':
            max_val = paramdict.get('max_val', np.max(X, axis))
            min_val = paramdict.get('min_val', np.min(X, axis))
            return (X - min_val) / (max_val - min_val), {'max_val': max_val, 'min_val': min_val}
        if norm == 'normal':
            from sklearn.preprocessing import StandardScaler
            return StandardScaler().fit_transform(X), dict()
        from sklearn.preprocessing import normalize
        return normalize(X, norm, axis), dict()

    @classmethod
    def calMissRate(cls, df, col):
        """
        计算某一列的缺失率
        :param df:
        :param col:
        :return:
        """
        r = df[df[col].isnull()].shape[0] / df.shape[0]
        print(f'字段{col}的缺失率为：{round(r, 2)}')
        return r

    @classmethod
    def calMissRateImpact2Labels(cls, df, col, labelCol, labels, verbose=True):
        """
        计算缺失率对2分类的影响
        :param df:
        :param col:
        :param labelCol:
        :param labels: list. [label1, label2]
        :return:
        """
        df_null = df[df[col].isnull()]
        a = df_null[labelCol].value_counts()
        if verbose:
            print(a)
        if 0 in a:
            r1 = 1.0 * a[labels[0]] / df_null.shape[0]
        else:
            r1 = 0
        print(f'特征 {col} 缺失时, label {labels[0]} 占总样本数比 = {round(r1, 2)}')
        df_notnull = df[df[col].notnull()]
        a = df_notnull[labelCol].value_counts()
        if verbose:
            print(a)
        if 0 in a:
            r2 = 1.0 * a[labels[0]] / df_notnull.shape[0]
        else:
            r2 = 0
        print(f'特征 {col} 非缺失时, label {labels[0]} 占总样本数比 = {round(r2, 2)}')
        return r1, r2



    @classmethod
    def hosRank(cls, hosRankSeries):
        """
        把医院的等级描述转化为相应的数字
        :param hosRankSeries: 医院等级的Series
        :return:
        """
        def t(x):
            if x.find('三级') != -1:
                return 3
            if x.find('二级') != -1:
                return 2
            if x.find('一级') != -1:
                return 1
            if x.find('未评级') != -1:
                return 0
        return hosRankSeries.apply(t)