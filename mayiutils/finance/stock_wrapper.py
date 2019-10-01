#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: stock_wrapper.py
@time: 2019-08-11 22:57
"""
from datetime import datetime, timedelta
import pandas as pd
import tushare as ts


def getColnames(s):
    """
    常见的copy网页上的表格时，处理利用
    s = \"""ts_code	str	TS股票代码
    trade_date	str	交易日期
    close	float	当日收盘价
    turnover_rate	float	换手率（%）
    turnover_rate_f	float	换手率（自由流通股）
    circ_mv	float	流通市值（万元）\"""
    :param s:
    :return:
        'ts_code,trade_date...,total_mv,circ_mv'
        ['TS股票代码', ...'流通市值（万元）']
    """
    return ','.join([i.split()[0] for i in s.split('\n')]), [i.split()[2] for i in s.split('\n')]
def df2dicts(df):
    """
    df to dicts list
    """
    dicts = []
    for line in df.itertuples():
        ll = list(df.columns)
        dicts.append(dict(zip(ll, list(line)[1:])))
    return dicts





def df2dicts_stock(df):
    """
    df to dicts list
    """
    dicts = []
    for line in df.itertuples():
        ll = ['trade_date'] + list(df.columns)
        dicts.append(dict(zip(ll, [line[0]]+list(line)[1:])))
    return dicts


class TushareWrapper:
    def __init__(self):
        TS_TOKEN = '5fd1639100f8a22b7f86e882e03192009faa72bae1ae93803e1172d5'
        self._pro = ts.pro_api(TS_TOKEN)

    def get_tushare_pro(self):
        return self._pro

    def get_stock_list(self):
        """
        查询当前所有正常上市交易的股票列表
        :return:
        """
        data = self._pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        return data

    def daily(self, ts_code, start_date, end_date, mode='index'):
        '''
        获取每日行情数据
        由于ts的接口一次只能获取1800个交易日（一年大概有250个交易日。约7年）的数据
        :mode
            index: 指数行情
            stock: 个股行情
        '''
        pro = self._pro
        startdate = datetime.strptime(start_date, '%Y%m%d')
        enddate = datetime.strptime(end_date, '%Y%m%d')
        df = pd.DataFrame()
        while enddate.year - startdate.year > 6:
            print(startdate.strftime('%Y%m%d'),
                  (startdate.replace(year=(startdate.year + 6)) - timedelta(days=1)).strftime('%Y%m%d'))
            params = {'ts_code': ts_code,
                      'start_date': startdate.strftime('%Y%m%d'),
                      'end_date': (startdate.replace(year=(startdate.year + 6)) - timedelta(days=1)).strftime('%Y%m%d')}
            if mode == 'index':
                t = pro.index_daily(**params)
            elif mode == 'stock':
                t = pro.daily(**params)
            elif mode == 'fund':
                t = pro.fund_daily(**params)

            if not df.empty:
                df = pd.concat([df, t], axis=0, ignore_index=True)
            else:
                df = t
            startdate = startdate.replace(year=(startdate.year + 6))
        else:
            print(startdate.strftime('%Y%m%d'), end_date)
            params = {'ts_code': ts_code,
                      'start_date': startdate.strftime('%Y%m%d'),
                      'end_date': end_date}

            if mode == 'index':
                t = pro.index_daily(**params)
            elif mode == 'stock':
                t = pro.daily(**params)
            elif mode == 'fund':
                t = pro.fund_daily(**params)

            if not df.empty:
                df = pd.concat([df, t], axis=0, ignore_index=True)
            else:
                df = t
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        return df