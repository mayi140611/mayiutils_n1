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
    def describe(cls, df):
        """
        描述df的
            data types
            percent missing
            unique values
            mode 众数
            count mode 众数计数
            % mode 众数占所有数据的百分比
            distribution stats  分布数据 分位数
        :param df:
        :return:
        """
        # data types
        dqr_data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])

        # percent missing
        dqr_percent_missing = pd.DataFrame(100 * (df.isnull().sum() / len(df)).round(3), columns=['% Missing'])

        # unique values
        dqr_unique_values = pd.DataFrame(columns=['Unique Values'])
        for c in df:
            dqr_unique_values.loc[c] = df[c].nunique()

        # mode 众数
        dqr_mode = pd.DataFrame(df.mode().loc[0])
        dqr_mode.rename(columns={dqr_mode.columns[0]: "Mode"}, inplace=True)

        # count mode
        dqr_count_mode = pd.DataFrame(columns=['Count Mode'])
        for c in df:
            dqr_count_mode.loc[c] = df[c][df[c] == dqr_mode.loc[[c]].iloc[0]['Mode']].count()

            # % mode
        dqr_percent_mode = pd.DataFrame(100 * (dqr_count_mode['Count Mode'].values / len(df)), \
                                        index=dqr_count_mode.index, columns=['% Mode'])

        # distribution stats
        df['temp_1a2b3c__'] = 1
        dqr_stats = pd.DataFrame(df['temp_1a2b3c__'].describe())
        del df['temp_1a2b3c__']
        for c in df:
            dqr_stats = dqr_stats.join(pd.DataFrame(df[c].describe()))
        del dqr_stats['temp_1a2b3c__']
        dqr_stats = dqr_stats.transpose().drop('count', axis=1)

        print("num of records: {}, num of columns: {}".format(len(df), len(df.columns)))

        return dqr_data_types.join(dqr_unique_values[['Unique Values']].astype(int)). \
            join(dqr_percent_missing).join(dqr_mode).join(dqr_count_mode[['Count Mode']].astype(int)).join(dqr_percent_mode).join(dqr_stats)

    @classmethod
    def normalize(cls, X, norm='01', axis=0, paramdict=dict()):
        """
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
    def build_one_hot_features(cls, df, cols):
        """
        构建one-hot特征
        :param df:
        :param cols: list
        :return:
        """
        for col in cols:
            t = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, t], axis=1)
            del df[col]
        return df

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