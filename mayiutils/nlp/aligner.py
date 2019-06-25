#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: aligner.py
@time: 2019-06-25 12:02
"""
from mayiutils.nlp.nlp_data_prepare import NLPDataPrepareWrapper as npw
import re
import os
from mayiutils.fileio.pickle_wrapper import PickleWrapper
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def standardize_diag(s):
    """
    字符串标准化
        去除所有空格
        去掉末尾最后一个 的
        小写转大写
        中文字符替换： （），【】：“”’‘；
    :param s:
    :return:
    """
    s = npw.standardize(s)
    s = re.sub(r'沙门氏菌', '沙门菌', s)
    return s


def standardize_mat(s):
    s = npw.standardize(s)
    s = re.sub(r'[3-9]+', '', s)  # 去除数字
    return s


def tokenizer(s):
    ll = npw.tokenizer(s, ['的'])
    return ' '.join(ll), ''.join(ll)


def diag_init():
    dis_name_codes_dict = PickleWrapper.loadFromFile(os.path.join(os.path.dirname(__file__), 'data/dis_name_codes_dict.pkl'))
    dis_code_name_dict = PickleWrapper.loadFromFile(os.path.join(os.path.dirname(__file__), 'data/dis_code_name_dict.pkl'))
    name_list, tfidf, tfidf_features = npw.buildTfidf(list(dis_name_codes_dict.keys()))
    return dis_name_codes_dict, dis_code_name_dict, name_list, tfidf, tfidf_features


def mat_init():
    print(os.path.dirname(__file__))
    # print(os.path.abspath('data/mat_common_products_dict.pkl'))
    # print(os.path.abspath('./data/mat_common_products_dict.pkl'))
    mat_common_products_dict = PickleWrapper.loadFromFile(os.path.join(os.path.dirname(__file__), 'data/mat_common_products_dict.pkl'))
    mat_product_common_dict = PickleWrapper.loadFromFile(os.path.join(os.path.dirname(__file__), 'data/mat_product_common_dict.pkl'))
    mat_name_list, mat_tfidf, mat_tfidf_features = npw.buildTfidf(list(mat_product_common_dict.keys()))
    return mat_common_products_dict, mat_product_common_dict, mat_name_list, mat_tfidf, mat_tfidf_features


mat_common_products_dict, mat_product_common_dict, mat_name_list, mat_tfidf, mat_tfidf_features = mat_init()


dis_name_codes_dict, dis_code_name_dict, name_list, tfidf, tfidf_features = diag_init()

class Aligner:
    def alignDiag(self, dfIn, sim=False, threhold=0.75):
        """

        :param dfIn:
            列名分别是diag_code和diag_name
        :param sim: 匹配度
        :param threhold:
            阈值，低于阈值返回none
        :return:
        """
        dfIn['diag_name1'] = dfIn['diag_name'].map(standardize_diag).map(tokenizer)
        dft = dfIn[dfIn['diag_name1'].str.get(1).isin(dis_name_codes_dict)].copy()  # 完全匹配
        dft['aligned_diag_code'] = dft['diag_name1'].str.get(1).map(lambda x: dis_name_codes_dict[x])
        dft['aligned_diag_name'] = dft['diag_name1'].str.get(1)
        dft['sim'] = 2
        dft2 = dfIn[dfIn['diag_name1'].str.get(1).isin(dis_name_codes_dict)==False].copy()  # 不完全匹配
        t = tfidf.transform(dft2['diag_name1'].str.get(0).to_list())
        start = datetime.now()
        r1 = cosine_similarity(t, tfidf_features)
        # 最相似的
        r2 = np.argmax(r1, axis=1)
        df = pd.DataFrame()
        df['diag_code'] = dft2['diag_code'].to_list()
        df['diag_name'] = dft2['diag_name'].to_list()
        df['aligned_diag_code'] = [dis_name_codes_dict[name_list[i]] for i in r2.tolist()]
        df['aligned_diag_name'] = [name_list[i] for i in r2.tolist()]
        df['sim'] = np.max(r1, axis=1)
        df.loc[df['sim'] < threhold, 'aligned_diag_code'] = None
        df.loc[df['sim'] < threhold, 'aligned_diag_name'] = None
        df = pd.concat([df, dft], ignore_index=True)
        df = df[['diag_code', 'diag_name', 'aligned_diag_code', 'aligned_diag_name', 'sim']].sort_values('sim', ascending=False)
        print(f'匹配数据条数: {dfIn.shape[0]}, 耗时{(datetime.now()-start).total_seconds()}s。\n/'
              f'未匹配条目数: {df.loc[df["sim"] < threhold].shape[0]}\n/'
              f'完全匹配条目数: {dft.shape[0]}')
        if not sim:
            del df['sim']
        return df

    def alignItem(self, dfIn, sim=False, threhold=0.75):
        """

        :param dfIn:
            列名item_name
        :param sim: 匹配度
        :param threhold:
            阈值，低于阈值返回none
        :return:
        """
        t = mat_tfidf.transform(dfIn['item_name'].map(standardize_mat).map(tokenizer).to_list())

        start = datetime.now()
        r1 = cosine_similarity(t, mat_tfidf_features)
        # 最相似的
        r2 = np.argmax(r1, axis=1)
        df = pd.DataFrame()
        df['item_name'] = dfIn['item_name'].to_list()
        df['aligned_name'] = [mat_name_list[i] for i in r2.tolist()]
        df['aligned_name_cat'] = [mat_product_common_dict[mat_name_list[i]] for i in r2.tolist()]
        df['sim'] = np.max(r1, axis=1)

        df.loc[df['sim'] < threhold, 'aligned_name'] = None
        df.loc[df['sim'] < threhold, 'aligned_name_cat'] = None

        print(f'匹配数据条数: {dfIn.shape[0]}, 耗时{(datetime.now()-start).total_seconds()}s。\n未匹配条目数: {df.loc[df["sim"] < threhold].shape[0]}')
        if not sim:
            del df['sim']
        return df