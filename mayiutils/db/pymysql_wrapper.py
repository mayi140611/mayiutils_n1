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
import pandas as pd


class PyMysqlWrapper:
    """
    常用是sql场景：
    查询不重复的记录：
        select distinct name from user;
        select id,name from user group by name;
    """
    def __init__(self, user='root', passwd='123456', db='daozhen', host='127.0.0.1', use_unicode=True, charset='utf8'):
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

    def executeAndCommit(self, query, args=None):
        """
        Execute a query 并且提交,返回查询结果条数
        :return:
        """
        n = self._cursor.execute(query, args)
        self._conn.commit()
        return n

    def excuteMany(self, query, args):
        """
        Run several data against one query  并且提交
        cursor.executemany("insert into tb7(user,pass,licnese)values(%s,%s,%s)",
            [("u3","u3pass","11113"),("u4","u4pass","22224")])
        :param query: query to execute on server
        :param args: Sequence of sequences or mappings.  It is used as parameter.
        :return: Number of rows affected, if any.
        """
        n = self._cursor.executemany(query, args)
        self._conn.commit()
        return n

    def close(self):
        self._cursor.close()
        self._conn.close()


if __name__ == '__main__':
    # pmw = PyMysqlWrapper(host='172.17.0.1', db='dics')
    pmw = PyMysqlWrapper(host='10.0.104.44', db='dics')
    mode = 1
    if mode == 2:
        sqltemplate = """
        INSERT INTO synonyms SET source = "{}", target = "{}"
        """
        df = pd.read_excel('../../tmp/syn.xlsx')
        # print(df.head())
        for line in df.itertuples():
            print(sqltemplate.format(line[1], line[2]))
            pmw.executeAndCommit(sqltemplate.format(line[1], line[2]))
    if mode == 1:
        sqltemplate = """
        SELECT *
        from disease_dic
        """
        pmw.execute(sqltemplate)
        data = pmw._cursor.fetchone()
        print(data)