#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: plot_img_ops.py
@time: 2019-11-18 11:47
"""
import matplotlib.pyplot as plt
import numpy as np
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot2D(func, xRange, xlabel='x', ylabel='f(x)'):
    """
    绘制2D函数图像
    e.g.
        plot2D(np.tanh, [-2, 2])
    :param func:
    :param xRange:
    :return:
    """
    xArr = np.linspace(xRange[0], xRange[1], num=1000)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xArr, func(xArr))
    plt.show()