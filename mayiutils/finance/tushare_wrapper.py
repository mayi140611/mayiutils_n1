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

    def get_stocks_info(self):
        """
        查询当前所有正常上市交易的股票列表
        :return:
        """
        data = self._pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        return data

    def daily(self, trade_date):
        """
        获取某一日全部股票的数据
        :param trade_date:
        :return:
        """
        return self._pro.daily(trade_date=trade_date)

    def history(self, ts_code, start_date, end_date, asset='E', adj='qfq'):
        """
        获取某只股票的历史行情数据
        由于ts的接口一次只能获取1800个交易日（一年大概有250个交易日。约7年）的数据
        :param ts_code:
        :param start_date: str
        :param end_date: str
        :param asset: 资产类别：E股票 I沪深指数 C数字货币 FT期货 FD基金 O期权 CB可转债（v1.2.39），默认E
        :param adj: 复权类型(只针对股票)：None未复权 qfq前复权 hfq后复权 , 默认None
        :return:
        """
        pro = self._pro
        startdate = datetime.strptime(start_date, '%Y%m%d')
        enddate = datetime.strptime(end_date, '%Y%m%d')
        df = pd.DataFrame()
        while enddate.year - startdate.year > 6:
            print(startdate.strftime('%Y%m%d'),
                  (startdate.replace(year=(startdate.year + 6)) - timedelta(days=1)).strftime('%Y%m%d'))
            params = {'ts_code': ts_code,
                      'start_date': startdate.strftime('%Y%m%d'),
                      'asset': asset,
                      'api': self._pro,
                      'end_date': (startdate.replace(year=(startdate.year + 6)) - timedelta(days=1)).strftime('%Y%m%d'),
                      'adj': adj}
            # if mode == 'index':
            #     t = pro.index_daily(**params)
            # elif mode == 'stock':
            #     t = pro.daily(**params)
            #
            # elif mode == 'fund':
            #     t = pro.fund_daily(**params)
            t = ts.pro_bar(**params)
            if not df.empty:
                df = pd.concat([df, t], axis=0, ignore_index=True)
            else:
                df = t
            startdate = startdate.replace(year=(startdate.year + 6))
        else:
            print(startdate.strftime('%Y%m%d'), end_date)

            params = {'ts_code': ts_code,
                      'start_date': startdate.strftime('%Y%m%d'),
                      'asset': asset,
                      'api': self._pro,
                      'end_date': end_date,
                      'adj': adj}
            # if mode == 'index':
            #     t = pro.index_daily(**params)
            # elif mode == 'stock':
            #     t = pro.daily(**params)
            # elif mode == 'fund':
            #     t = pro.fund_daily(**params)
            t = ts.pro_bar(**params)
            if not df.empty:
                df = pd.concat([df, t], axis=0, ignore_index=True)
            else:
                df = t
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        return df

    def get_daily_basic(self, t_date):
        f = 'ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv'
        df = self._pro.daily_basic(ts_code='', trade_date=t_date, fields=f)
        df.columns = [
            'ts_code',
            'trade_date',
            'close',
            '换手率（%）',
            '换手率（自由流通股）',
            '量比',
            '市盈率（总市值/净利润）',
            '市盈率（TTM）',
            '市净率（总市值/净资产）',
            '市销率',
            '市销率（TTM）',
            '总股本',
            '流通股本',
            '自由流通股本',
            '总市值',
            '流通市值（万元）'
        ]
        if not df.empty:
            print(f'请求到{len(df)}条数据！')
            df.trade_date = pd.to_datetime(df.trade_date)
            df = df.sort_values('总市值', ascending=False)
            df['rank'] = range(1, df.shape[0] + 1)
        return df
