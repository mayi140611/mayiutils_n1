#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: time_utils.py
@time: 2019-08-12 18:57
"""
from datetime import datetime


def oracle_time_transform(s):
    """
    '22-9月 -13'
    :param d:
    :return:
    """
    def t(a):
        return a if len(a)>1 else '0'+a
    try:
        ss = [i.strip() for i in s.split('-')]
        ss[1] = ss[1][:-1]
        ss = list(map(t, ss))
    except Exception as e:
        print(str(e), s)
        return datetime.strptime('19000101', '%Y%m%d')
    return datetime.strptime(f'{ss[2]}{ss[1]}{ss[0]}', '%y%m%d')


if __name__ == '__main__':
    print(oracle_time_transform('22-9月 -13'))