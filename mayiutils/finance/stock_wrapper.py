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


def df2dicts(df):
    """
    df to dicts list
    """
    dicts = []
    for line in df.itertuples():
        ll = list(df.columns)
        dicts.append(dict(zip(ll, list(line)[1:])))
    return dicts


def get_tushare_pro():
    TS_TOKEN = '5fd1639100f8a22b7f86e882e03192009faa72bae1ae93803e1172d5'
    pro = ts.pro_api(TS_TOKEN)
    return pro


def df2dicts_stock(df):
    """
    df to dicts list
    """
    dicts = []
    for line in df.itertuples():
        ll = ['trade_date'] + list(df.columns)
        dicts.append(dict(zip(ll, [line[0]]+list(line)[1:])))
    return dicts


def daily(ts_code, start_date, end_date, mode='index'):
    '''
    获取每日行情数据
    由于ts的接口一次只能获取1800个交易日（一年大概有250个交易日。约7年）的数据
    :mode
        index: 指数行情
        stock: 个股行情
    '''
    pro = get_tushare_pro()
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