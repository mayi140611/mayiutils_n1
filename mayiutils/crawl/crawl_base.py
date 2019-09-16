#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: crawl_base.py
@time: 2019-09-16 15:45

爬虫基类
"""
import requests
from pyquery import PyQuery as pq
from requests import RequestException


class Crawl(object):
    @classmethod
    def get_page_index(cls, url, encoding='utf8', headers=None):
        if not headers:
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/61.0.3163.79 Safari/537.36 Maxthon/5.2.1.6000',
            }
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                r.encoding = encoding
                return r.text
        except RequestException:
            print(f'get_page_index {url} Error occurred')
        return None

    @classmethod
    def parse_page_index(cls, html):
        root = pq(html)
        pass
        detail_url_list = []
        return detail_url_list

    @classmethod
    def get_page_detail(cls, url, encoding='utf8', headers=None):
        return cls.get_page_index(url, encoding, headers)

    @classmethod
    def parse_page_detail(cls, url, html):
        root = pq(html)
        d = dict()
        d['url'] = url
        pass
        return d

    @classmethod
    def main(cls, url, save_path):
        html = cls.get_page_index(url, encoding='utf8', headers=None)
        detail_url_list = cls.parse_page_index(html)
        from pandas import Series
        Series(detail_url_list).to_pickle(save_path)  # 落盘
        result_list = []
        start = 0
        for i in range(start, len(detail_url_list)):
            print(f'{i} {detail_url_list[i]}')
            html = cls.get_page_detail(detail_url_list[i], encoding='utf8', headers=None)
            result_list.append(cls.parse_page_detail(detail_url_list[i], html))
        # import sys
        # sys.path.append('/Users/luoyonggui/PycharmProjects/mayiutils_n1/mayiutils/db')
        # from pymongo_wrapper import PyMongoWrapper
        # mongo = PyMongoWrapper()
