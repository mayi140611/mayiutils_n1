#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: hive_wrapper.py
@time: 2019-08-06 13:49
"""
from pyhive import hive


def connect_hive(host, port=10000, database='default', username='hive'):
    try:
        conn = hive.Connection(host=host, port=port, database=database, username=username)
        cursor = conn.cursor()
        # cursor.execute('create table if not exists scores_model(model_file string,create_time int)')
        print('hive连接成功！')
        return conn, cursor
    except Exception as e:
        #     raise Exception("hive连接异常！")
        print('hive连接失败！')