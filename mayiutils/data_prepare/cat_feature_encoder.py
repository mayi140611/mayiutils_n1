#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: cat_feature_encoder.py
@time: 2019-12-05 17:36
"""
import pandas as pd


class CatFeatureEncoder:
    """类别特征编码"""

    @classmethod
    def build_one_hot_features(cls, df, cols: list):
        """
        构建one-hot特征
        可以处理None值：会被处理为全0
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












