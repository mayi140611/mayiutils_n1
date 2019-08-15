#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pymysql_wrapper.py
@time: 2019/1/30 18:36
PyMySQL==0.9.2
pip install pymysql
"""
import pymysql


class PyMysqlWrapper:
    """
    常用是sql场景：
    查询不重复的记录：
        select distinct name from user;
        select id,name from user group by name;
    """
    def __init__(self, user='root', passwd='123456', db='test', host='127.0.0.1', use_unicode=True, charset='utf8'):
        """
        创建连接、创建游标
        :param user:
        :param passwd:
        :param db:
        :param host:
        :param use_unicode:
        :param charset:
        :return:
        """
        self._conn = pymysql.connect(user=user, passwd=passwd, db=db, host=host, use_unicode=use_unicode, charset=charset)
        self._cursor = self._conn.cursor()

    def execute(self, query, args=None):
        """
        Execute a query 并且提交,返回查询结果条数
        :return:
        """
        n = self._cursor.execute(query, args)
        return n

    def excuteMany(self, query, args):
        """

        """
        n = self._cursor.executemany(query, args)
        return n

    def commit(self, query, args):
        """
        """
        self._conn.commit()

    def close(self):
        self._cursor.close()
        self._conn.close()
