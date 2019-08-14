#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: logging_utils.py
@time: 2019-08-14 10:30
"""
import logging
import sys


def get_logger(module_name, stream_handler=True, file_handler=False, log_path=''):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # StreamHandler
    if stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    # FileHandler
    if file_handler:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
