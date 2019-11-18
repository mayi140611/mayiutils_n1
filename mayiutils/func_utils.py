#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: func_utils.py
@time: 2019-11-18 11:55

常见的函数及其实现
"""
import numpy as np


def sigmoid(z):
    """
    sigmoid function or logistic function is used as a hypothesis function in classification problems
    """
    return 1 / (1 + np.exp(-z))


def softmax(x):
    """

    :param x: list
    :return:
    """
    arr1d = np.array(x)
    exp = np.exp(arr1d - arr1d.max())  # 防止指数爆炸
    return exp / np.sum(exp)


def loss_function(y_pred, y_true):
    """
    逻辑回归专用代价函数
    """
    return (-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)).mean()


if __name__ == "__main__":
    print(softmax([1, 2, 3, 4]))