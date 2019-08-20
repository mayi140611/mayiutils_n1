#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: hdfs_wrapper.py
@time: 2019-08-19 19:39
"""


def download_parquet_from_hdfs(parquet_path, local_path, hdfs_ip, hdfs_port=50070):
    """
    从hdfs下载parquet文件到local_path
    :param parquet_path: '/data/a.parquet'
    :param local_path: '/data_gen/b.parquet'
    :param hdfs_ip:
    :param hdfs_port:
    :return:
    """
    from hdfs.client import Client
    client = Client(f'http://{hdfs_ip}:{hdfs_port}')
    with client.read(parquet_path) as reader:
        data = reader.read()
    with open(local_path, 'wb') as f:
        f.write(data)


def download_parquet_from_hdfs_dir(parquet_dir, local_dir, hdfs_ip, hdfs_port=50070):
    """
    从hdfs批量下载parquet文件到local_path
    :param parquet_dir: parquet文件所在的文件'/data/a.parquet'
    :param local_path: '/data_gen/b.parquet'
    :param hdfs_ip:
    :param hdfs_port:
    :return:
    """
    import os
    from hdfs.client import Client
    client = Client(f'http://{hdfs_ip}:{hdfs_port}')
    parquet_list = client.list(parquet_dir)
    print(parquet_list)
    for p in parquet_list:
        if p.endswith('.parquet'):
            print(f'downloading {os.path.join(parquet_dir, p)}')
            with client.read(os.path.join(parquet_dir, p)) as reader:
                data = reader.read()
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            with open(os.path.join(local_dir, p), 'wb') as f:
                f.write(data)
