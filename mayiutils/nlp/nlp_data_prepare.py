#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: nlp_data_prepare.py
@time: 2019-05-05 14:56

自然语言处理分为如下几个步骤：
0、先对语料进行规范化
    常见的规范化有：
        去除停用词
        统一大小写
        统一中英文符号
1、word/char 利用语料构建字典
    word_index
    index_word
2、利用字典对语料进行编码/解码
    encode(text):
        :return seq
    decode(seq):
        :return text
3、把语料进行填补至等长 padding

"""
import collections
import re
import os
import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全
pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行


class NLPDataPrepareWrapper:
    @classmethod
    def buildDataset(cls, words, vocabulary_size):
        """

        :param words: 所有文章分词后的一个words list
        :param vocabulary_size: 取频率最高的词数
        :return:
            data 编号列表，编号形式
            count 前50000个出现次数最多的词
            dictionary 词对应编号
            reverse_dictionary 编号对应词
        """
        count = [['UNK', -1]]
        # 前50000个出现次数最多的词
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        # 生成 dictionary，词对应编号, word:id(0-49999)
        # 词频越高编号越小
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        # data把数据集的词都编号
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        # 记录UNK词的数量
        count[0][1] = unk_count
        # 编号对应词的字典
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary

    @classmethod
    def standardize(cls, s):
        """
        字符串标准化
            去除两边的空格
            中文字符替换： （），【】：“”’‘；?
        :param s:
        :return:
        """
        if type(s) != str:
            s = str(s)
        s = s.strip()
        # s = re.sub(r'\s+', ' ', s)
        s = s.lower()
        s = re.sub(r'（', '(', s)
        s = re.sub(r'）', ')', s)
        s = re.sub(r'，', ',', s)
        s = re.sub(r'：', ':', s)
        s = re.sub(r'【', '[', s)
        s = re.sub(r'】', ']', s)
        s = re.sub(r'“|”|’|‘', '"', s)
        s = re.sub(r'；', ';', s)
        s = re.sub(r'？', '?', s)
        return s

    @classmethod
    def tokenizer(cls, s, stopwords=None):
        """

        :param s:
        :param stopwords: list. 停用词列表
        :return:
        """
        ll = jieba.lcut(s)
        if stopwords:
            ll = [w for w in ll if w not in stopwords]
        return ll

    @classmethod
    def buildTfidf(cls, texts):
        tw = [' '.join(i) for i in map(cls.tokenizer, texts)]
        print('load stopwords')
        with open(os.path.join(os.path.dirname(__file__), 'data/stopwords_zh.dic'), encoding='utf8') as f:
            stopwords = [s.strip() for s in f.readlines()]
        print('building tfidf array')
        tfidf = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b\w+\b")
        tfidf.fit(tw)
        tfidf_features = tfidf.transform(tw)
        print('building tfidf array completed')
        return tw, tfidf, tfidf_features

    @classmethod
    def align(cls, baseCol, col, isunique=False):
        """
        把一列数据匹配到基准列
        :param baseCol: Series
        :param col: Series
        :return:
        """
        if isunique:
            col = pd.Series(col.unique())
            baseCol = pd.Series(baseCol.unique())
        col1 = col.map(cls.standardize)
        baseCol1 = baseCol.map(cls.standardize)
        df1 = pd.DataFrame({'col': col,
                            'sss': col1})
        df2 = pd.DataFrame({'baseCol': baseCol,
                            'sss': baseCol1})
        # 处理完全匹配
        dfr1 = pd.merge(df1, df2, how='inner', on='sss')
        dfr1['sim'] = 2
        # 处理非完全匹配
        col2 = col[col1.isin(dfr1['sss'])==False]
        tw, tfidf, tfidf_features = cls.buildTfidf(baseCol1.to_list())
        t = tfidf.transform(col2.map(cls.standardize).map(cls.tokenizer).map(lambda s: ' '.join(s)))
        r1 = cosine_similarity(t, tfidf_features)
        # 最相似的
        r2 = np.argmax(r1, axis=1)
        dfr2 = pd.DataFrame()
        dfr2['col'] = col2.to_list()
        dfr2['baseCol'] = [baseCol[i] for i in r2.tolist()]
        dfr2['sim'] = np.max(r1, axis=1)
        df = pd.concat([dfr1, dfr2], ignore_index=True)
        df = df[['col', 'baseCol', 'sim']].sort_values('sim', ascending=False)
        print(f'一共匹配{len(col)}条，完全匹配：{dfr1.shape[0]}条')
        return df