#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: base.py
@time: 2019-09-16 16:51
"""


class Publisher:
    def __init__(self):
        pass

    def register(self):
        pass

    def unregister(self):
        pass

    def notify_all(self):
        pass


class Subscriber:
    def __init__(self):
        pass

    def notify(self):
        pass

from attr import attrs, attrib, fields
from cattr import unstructure, structure
from datetime import datetime


@attrs
class Transaction:
    num = attrib(type=float, default=0)
    price = attrib(type=float, default=0)
    tdatetime = attrib(type=datetime, default=datetime.now())
    target = attrib(type=str, default='cash')
    account = attrib(type=str, default='pingan')


# @attrs
# class Position:
#     target = attrib(type=str)
#     num = attrib(type=float, default=0)
#     price_mean = attrib(type=float, default=0)

@attrs
class Trade(Publisher):
    _transaction_list = attrib(factory=list)
    _accounts = attrib(factory=list)
    # def __init__(self, transaction_list):
    #     self._accounts = []
    #     self._transaction_list = transaction_list

    def register(self, account):
        if account not in self._accounts:
            self._accounts.append(account)

    def unregister(self, account):
        if account not in self._accounts:
            self._accounts.remove(account)

    def notify_all(self):
        for a in self._accounts:
            a.notify(self._transaction_list)


@attrs
class Account(Subscriber):
    _name = attrib(type=str)
    # _trade_list = attrib(type=list, default=[])
    # _trade_list = attrib(type=list, default=[])
    _trade_list = attrib(factory=list, repr=False)
    _positions = attrib(factory=dict)
    _cash = attrib(type=float, default=0)
    # def __init__(self, name):
    #     self._name = name
    #     self._trade_list = []
    #     self._positions = dict()  # target: (num, price_mean)
    #     self._cash = 0
    #
    # def __str__(self):
    #     return f'{self._name}: ' \
    #            f'{self._positions}\n{self._cash}'

    def notify(self, transaction_list):
        for t in transaction_list:
            if t.account == self._name:
                self._trade_list.append(t)
                if t.target == 'cash':
                    self._cash += t.num
                else:
                    num = t.num
                    amount = t.num * t.price
                    fee = 5 if amount*0.00025 < 5 else amount*0.00025
                    if t.target in self._positions:
                        num += self._positions[t.target][0]
                        amount += self._positions[t.target][1] * self._positions[t.target][0]
                    price_mean = (amount + fee)/num
                    self._positions[t.target] = (num, price_mean)
                print(self)


if __name__ == '__main__':
    transaction_list = []
    transaction = Transaction(90000, target='cash', account='pingan')
    transaction_list.append(transaction)
    transaction = Transaction(1000, price=0.844, target='军工ETF', account='ZLT')
    transaction_list.append(transaction)
    transaction = Transaction(90000, target='cash', account='ZLT')
    transaction_list.append(transaction)
    transaction = Transaction(1000, price=0.849, target='军工ETF', account='ZLT')
    transaction_list.append(transaction)

    trade = Trade(transaction_list)
    zlt = Account('ZLT')
    pingan = Account('pingan')
    trade.register(zlt)
    trade.register(pingan)
    trade.notify_all()

    print(unstructure(trade))

