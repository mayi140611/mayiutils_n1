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
    def standardize(cls, s, lower=True, whitespacereplace=False, rmParentheses=False, rmchar=False):
        """
        字符串标准化
            去除两边的空格
            中文字符替换： （），【】：“”’‘；?
        :param s:
        :param whitespacereplace: 是否移把多个空白字符替换为单个空格
        :param rmParentheses: 是否移除小括号及内部的内容
        :param rmchar: 是否移除英文字符
        :return:
        """
        if type(s) != str:
            s = str(s)
        s = s.strip()
        if whitespacereplace:
            s = re.sub(r'\s+', ' ', s)
        if rmchar:
            s = re.sub(r'[a-zA-Z]', '', s)
        elif lower:
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
        if rmParentheses:
            s = re.sub(r'\(.*?\)', '', s)
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
    def buildTfidf(cls, texts, pattern='word'):
        """
        构建tfidf矩阵，之前需要先进行标准化，但是不需要分词
        :param texts:
        :param pattern:
            char: 以字/char为单位，作为最小粒度。不分词，不去除停用词！
            word: 分词，去除停用词
        :return:
        """
        if pattern == 'char':
            tw = [' '.join(list(re.sub(r'\s+', '', i))) for i in texts]
            stopwords = None
        elif pattern == 'word':
            tw = [' '.join(i) for i in map(cls.tokenizer, texts)]
            print('load stopwords')
            with open(os.path.join(os.path.dirname(__file__), 'data/stopwords_zh.dic'), encoding='utf8') as f:
                stopwords = [s.strip() for s in f.readlines()]
        print('building tfidf array')
        tfidf = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b[\w\.\+\-/]+\b")  # 匹配字母数字下划线和小数点  如10.0
        tfidf.fit(tw)
        tfidf_features = tfidf.transform(tw)
        print('building tfidf array completed')
        return tw, tfidf, tfidf_features

    @classmethod
    def align(cls, baseCol, col, tfidfpattern='word'):
        """
        把一列数据匹配到基准列
        :param baseCol: Series
        :param col: Series
        :return:
        """
        col1 = col.map(cls.standardize)
        baseCol1 = baseCol.map(cls.standardize)
        df1 = pd.DataFrame({'col': col,
                            'sss': col1})
        df2 = pd.DataFrame({'baseCol': baseCol,
                            'sss': baseCol1})
        df1.drop_duplicates('sss', inplace=True)
        df2.drop_duplicates('sss', inplace=True)

        # 处理完全匹配
        dfr1 = pd.merge(df1, df2, how='inner', on='sss')
        dfr1['sim'] = 2
        # 处理非完全匹配
        col2 = col[col1.isin(dfr1['sss']) == False]
        tw, tfidf, tfidf_features = cls.buildTfidf(baseCol1.to_list(), pattern=tfidfpattern)
        if tfidfpattern == 'word':
            t = tfidf.transform(col2.map(cls.standardize).map(cls.tokenizer).map(lambda s: ' '.join(s)))
        elif tfidfpattern == 'char':
            t = tfidf.transform(col2.map(cls.standardize).map(lambda s: [i for i in s if i.strip()]).map(lambda s: ' '.join(s)))
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

    @classmethod
    def match(cls, df_base, base_name_col, df_target, target_name_col, target_file, order='keep'):
        """
        通用匹配，按照名称进行匹配，
        把df_target的target_name_col匹配到df_base的base_name_col
        :param df_base:
        :param base_name_col:
        :param df_target:
        :param target_name_col:
        :param target_file:
            输出到f"data_gen/{target_file.split('/')[-1].split('.')[0]}_匹配结果.xlsx"
        :param order:
            keep: 保持原有顺序；
            sim: 按照sim值从高到低排序
        :return:
        """
        def t(s):
            return cls.standardize(s, True, True, True)

        df_base['标准列'] = df_base[base_name_col].map(t)
        df_base.drop_duplicates('标准列', inplace=True)
        df = df_target
        df['_id'] = range(df.shape[0])
        print(f'一共待匹配条数: {df.shape[0]}')
        df['标准列'] = df[target_name_col].map(t)
        df.drop_duplicates('标准列')
        print(f'去重后，一共待匹配条数: {df.drop_duplicates("标准列").shape[0]}')
        df_exactmatch = pd.merge(df, df_base, on='标准列')
        df_exactmatch['sim'] = 2
        print(f'完全匹配条数: {df_exactmatch.shape}')
        df_base_o = df_base[df_base['标准列'].isin(df_exactmatch['标准列'].to_list()) == False]
        df_o = df[df['标准列'].isin(df_exactmatch['标准列'].to_list()) == False]
        print(df_o.shape)
        if df_o.shape[0] > 0:
            dft = cls.align(df_base_o.loc[:, '标准列'].reset_index(drop=True), df_o.loc[:, '标准列'].reset_index(drop=True),
                            tfidfpattern='char')
            print(dft.shape)
            df_o1 = pd.merge(df_o, dft, left_on='标准列', right_on='col')
            del df_o1['col']
            df_o1 = pd.merge(df_o1, df_base_o, left_on='baseCol', right_on='标准列')
            df_o1.drop(columns=['标准列_x', 'baseCol', '标准列_y'], inplace=True)
        else:
            df_o1 = pd.DataFrame()
        df_exactmatch.drop(columns=['标准列'], inplace=True)
        if order == 'sim':
            dfr = pd.concat([df_exactmatch, df_o1], ignore_index=True).sort_values('sim', ascending=False)
        else:
            dfr = pd.concat([df_exactmatch, df_o1], ignore_index=True).sort_values('_id')
        dfr = dfr[df_exactmatch.columns].drop_duplicates('_id')
        del dfr['_id']
        dfr.to_excel(f"data_gen/{target_file.split('/')[-1].split('.')[0]}_匹配结果.xlsx", index=False)
        return dfr

    @classmethod
    def match_illness(cls, df_base, base_code_col, base_name_col, df_target, target_code_col, target_name_col, target_file=None, order='keep'):
        """
        疾病匹配，先通过code匹配，匹配上的sim=3，然后在通过名称匹配
        :param df_base:
        :param base_code_col:
        :param base_name_col:
        :param df_target:
        :param target_code_col:
        :param target_name_col:
        :param target_file:
            输出到f"data_gen/{target_file.split('/')[-1].split('.')[0]}_匹配结果.xlsx"
        :return:
        """
        def t(s):
            return cls.standardize(s, True, True)

        print(df_base.head())
        df_base['标准列'] = df_base[base_code_col].map(t)
        df_base.drop_duplicates('标准列', inplace=True)
        df = df_target
        df['_id'] = range(df.shape[0])
        print(f'一共待匹配条数: {df.shape[0]}')
        print(df.head())
        df['标准列'] = df[target_code_col].map(t)

        print(f'去重后，一共待匹配条数: {df.drop_duplicates("标准列").shape[0]}')
        # 完全匹配

        df_exact = pd.merge(df, df_base, on=['标准列'])
        df_exact['sim'] = 3
        print(f'完全匹配条数: {df_exact.shape}')
        # 非完全匹配
        print('非完全匹配')
        df_base_o = df_base[df_base['标准列'].isin(df_exact['标准列'].to_list()) == False].copy()
        df_base_o['标准列'] = df_base_o[base_name_col].map(t)
        df_base_o.drop_duplicates('标准列', inplace=True)
        df_o = df[df['标准列'].isin(df_exact['标准列'].to_list()) == False].copy()
        if df_o.shape[0] > 0:
            df_o['标准列'] = df_o[target_name_col].map(t)
            dft = cls.align(df_base_o.loc[:, '标准列'].reset_index(drop=True), df_o.loc[:, '标准列'].reset_index(drop=True),
                            tfidfpattern='char')
            df_o1 = pd.merge(df_o, dft.drop_duplicates('col'), left_on='标准列', right_on='col')
            del df_o1['col']
            df_o1 = pd.merge(df_o1, df_base_o, left_on='baseCol', right_on='标准列')
            df_o1.drop(columns=['标准列_x', 'baseCol', '标准列_y'], inplace=True)
        else:
            df_o1 = pd.DataFrame()
        df_exact.drop(columns=['标准列'], inplace=True)

        if order == 'sim':
            dfr = pd.concat([df_exact, df_o1], ignore_index=True).sort_values('sim', ascending=False)
        else:
            dfr = pd.concat([df_exact, df_o1], ignore_index=True).sort_values('_id')
        dfr = dfr[df_exact.columns]
        if target_file:
            dfr.to_excel(f"data_gen/{target_file.split('/')[-1].split('.')[0]}_疾病匹配结果.xlsx", index=False)
        return dfr


if __name__ == '__main__':
    s = 'ab打分，c（好放射费)，fdf方（好放射费)，法发顺丰'
    r = NLPDataPrepareWrapper.standardize(s, rmParentheses=True, rmchar=True)
    print(r)
