#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: read_config_file.py
@time: 2019-08-06 13:41
"""


def read_config_file(fp: str, mode='r', encoding='utf8', prefix='#') -> dict:
    """
    读取文本文件，忽略空行，忽略prefix开头的行，返回字典
    :param fp: 配置文件路径
    :param mode:
    :param encoding:
    :param prefix:
    :return:
    """
    with open(fp, mode, encoding=encoding) as f:
        ll = f.readlines()
        ll = [i for i in ll if all([i.strip(), i.startswith(prefix) == False])]
        params = {i.split('=')[0].strip(): i.split('=')[1].strip() for i in ll}
    print(params)
    return params