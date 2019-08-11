#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: hive_wrapper.py
@time: 2019-08-06 13:49
"""
from pyhive import hive


def connect_hive(host, port=10000, database='default', username='hive', tablename='scores_model'):
    try:
        conn = hive.Connection(host=host, port=port, database=database, username=username)
        cursor = conn.cursor()
        if tablename == 'scores_model':
            cursor.execute('create table if not exists scores_model(company_code string, model_file string,create_time int)')
        elif tablename == 'case_feedback':  # 案件反馈
            cursor.execute('create table if not exists case_feedback(company_code string, claim_id string, rules_fired string, scores float, hit tinyint, flag tinyint, create_time int)')
        print(f'hive表{tablename} 连接成功！')
        return conn, cursor
    except Exception as e:
        print(f'hive连接失败！{str(e)}')