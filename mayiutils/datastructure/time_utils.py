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
        return a if len(a) > 1 else '0' + a

    try:
        ss = [i.strip() for i in s.split('-')]
        ss[1] = ss[1][:-1]
        ss = list(map(t, ss))
        year = '20' + ss[2] if int(ss[2]) < 30 else '19' + ss[2]
    except Exception as e:
        print(str(e), s)
        return None

    return datetime.strptime(f'{year}{ss[1]}{ss[0]}', '%Y%m%d')


if __name__ == '__main__':
    print(oracle_time_transform('22-9月 -13'))  # 2013-09-22 00:00:00
    print(oracle_time_transform('24-9月 -54'))  # 1954-09-24 00:00:00
