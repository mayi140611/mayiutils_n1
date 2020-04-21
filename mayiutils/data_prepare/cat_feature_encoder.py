#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: cat_feature_encoder.py
@time: 2019-12-05 17:36
"""
import pandas as pd
import numpy as np


class CatFeatureEncoder:
    """类别特征编码"""

    @classmethod
    def build_one_hot_features(cls, df, cols: list, mode='cols'):
        """
        构建one-hot特征
        可以处理None值：会被处理为全0
        :param df:
        :param cols: list
        :param mode:
            cols 一个one-hot占一列
            list 只返回一列，值为one-hot编码后的list
        :return:
        """
        df = df.copy()
        for col in cols:
            t = pd.get_dummies(df[col], prefix=col)
            if mode == 'cols':
                df = pd.concat([df, t], axis=1)
                del df[col]
            elif mode == 'list':
                dft = pd.Series()
                for i in df.index:
                    dft.loc[i] = t.loc[i].tolist()
                df[col] = dft

        return df

    @classmethod
    def ordinal_coding(cls, s: pd.Series, cats_ordinal: list):
        """
        顺序编码
        :param s:
        :param cats_ordinal: 顺序列表
        :return:
        """
        d = dict(zip(cats_ordinal, range(len(cats_ordinal))))
        return s.map(lambda x: d[x], na_action='ignore')

    @classmethod
    def label_encoder(cls, s: pd.Series):
        """
        我们也可以使用标签编辑器将变量编码为数字。
        标签编辑器本质上做的是它看到列中的第一个值并将其转换成0，下一个值转换成1，依次类推。
        这种方法在树模型中运行得相当好，当我在分类变量中有很多级别时，可以使用它。

        有空值会报错：TypeError: argument must be a string or number

        注意：可以做逆向变换：le.inverse_transform([0, 1, 2, 0])
        :param s:
        :return:
        """
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(s)
        return le.transform(s), le

    @classmethod
    def target_encode(cls, features, cat_cols, target_col, drop=True, base=True, alpha=0.1, prior=0.05, r=2):
        """
        分类任务才能使用target_encode。
        回归任务需要先对label分组成bucket，才能使用target_encode，参考Catboost
        :param features:
        :param cat_cols:
        :param target_col:
        :param drop:
        :param base:
        :param alpha:
        :param prior:
        :param r:
        :return:
        """
        df = features.copy()

        def te_func(count, total_count, base=base, alpha=alpha, prior=prior, r=r):
            """

            :param count:
            :param total_count:
            :param base: 基础算法，只算概率，不考虑惩罚项
            :param prior: 先验参数，防止被编码为0(编码0可以留给从未出现的特征类别!!!)
            :param r: 保留小数位数
            :return:
            """
            if base:
                return round(count / total_count, r)
            if total_count > 100:
                penalty = 1  # 不惩罚
            else:
                penalty = 1/(1+np.exp(-total_count*alpha))
            return round((count+prior)/(total_count+1)*penalty, r)
        for c in cat_cols:
            for feature_cls, total_count in df[c].value_counts().items():
                count = 0
                df1 = df.loc[df[c]==feature_cls]
                for i, count_i in df1[target_col].value_counts().items():
                    # if count == 0:  # 放弃对第一列编码
                    #     count += 1
                    #     continue
                    te_val = te_func(count_i, total_count)
                    df.loc[(df[c]==feature_cls), f'{c}_{i}']=te_val
            if drop:
                del df[c]
        return df.fillna(0)


