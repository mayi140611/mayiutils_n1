#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: fileoperation_wrapper.py
@time: 2019/2/21 10:14

参考链接：https://docs.python.org/3.6/library/io.html
"""
from six.moves import urllib
import os
import zipfile
import tensorflow as tf


class FileOperationWrapper:
    """
    文件访问模式：
        r：以只读方式打开文件，文件指针放在开头
        rb：以二进制格式打开一个文件用于只读
        r+: 打开一个文件用于读写
        rb+
        w: 打开一个文件只用于写入，如果文件存在，则覆盖，如果不存在，创建新文件
        wb
        w+
        wb+
        a: 打开一个文件用于追加，文件指针放在末尾。如果文件不存在，创建新文件
        ab
        a+
        ab+
    """
    @classmethod
    def read(cls, f, size=-1):
        """
        Read up to size bytes from the object and return them.
        As a convenience, if size is unspecified or -1, all bytes until EOF are returned.
        :param f:
        :return:
        """
        return f.read()

    @classmethod
    def readZipFile(cls, filepath):
        """

        :param filepath:
        :return:
        """
        with zipfile.ZipFile(filepath) as f:
            # f.namelist() 返回所有文件夹和文件
            data = f.read(f.namelist()[2]).decode('utf8')
        return data

    @classmethod
    def writeList2File(cls, list, filepath, encoding='utf8'):
        """
        把list写入文件
        :param list:
        :param filepath:
        :param encoding:
        :return:
        """
        with open(filepath, 'w+', encoding=encoding) as f:
            f.writelines(list)

    @classmethod
    def downloadFromWeb(cls, url, filepath, expected_bytes=-1):
        """
        Retrieve a URL into a temporary location on disk if not present, and make sure it's the right size.
        :param url: 'http://mattmahoney.net/dc/text8.zip'
        :param filepath: 文件硬盘存放目录
        :param expected_bytes:
        :return:
        """
        if not os.path.exists(filepath):
            print('downloading from {}'.format(url))
            urllib.request.urlretrieve(url, filepath)
        else:
            print('{} 已存在，不需要下载')
        # 获取文件相关属性
        statinfo = os.stat(filepath)
        # 比对文件的大小是否正确
        if expected_bytes != -1:
            if statinfo.st_size == expected_bytes:
                print('Found and verified', filepath)
            else:
                print(statinfo.st_size)
                raise Exception(
                    'Failed to verify ' + filepath + '. Can you get to it with a browser?')
        return filepath

    @classmethod
    def downloadFromWeb2(cls, url, filename, filedir):
        """
        Downloads a file from a URL if it not already in the cache.
        :return:
        """
        return tf.keras.utils.get_file(fname=filename, origin=url, cache_dir=filedir, cache_subdir='')


if __name__ == '__main__':
    mode = 5
    if mode == 5:
        """
        """
        print(FileOperationWrapper.readZipFile('/Users/luoyonggui/Downloads/data.zip'))

    if mode == 4:
        """
        把文件按行读取为list
        """
        f = open('../nlp/jieba_userdict.txt', encoding='utf8')
        # s = f.readlines()
        # print(s)
        """
        ['云计算 5\n', '李小福 2 nr\n', '我爱李小 3\n', '创新办 3 i\n', 'easy_install 3 eng\n', '好用 300\n', '韩玉赏鉴 3 nz\n', '八一双鹿 3 nz\n', '台中\n']
        """
        print([i.strip() for i in f.readlines()])
        """
        ['云计算 5', '李小福 2 nr', '我爱李小 3', '创新办 3 i', 'easy_install 3 eng', '好用 300', '韩玉赏鉴 3 nz', '八一双鹿 3 nz', '台中']
        """
        f.close()
    if mode == 3:
        # 打开一个文件
        fo = open("foo.txt", "r+")
        print(type(fo))#<class '_io.TextIOWrapper'>
        str = fo.read(10);
        print("读取的字符串是 : ", str)
        # 关闭打开的文件
        fo.close()
    if mode == 2:
        fo = open("foo.txt", "a")
        print("文件名: ", fo.name)#foo.txt
        print("是否已关闭 : ", fo.closed)#False
        print("访问模式 : ", fo.mode)#a
        fo.write("1www.runoob.com!\nVery good site!\n");
        fo.close()
    if mode == 1:
        # 键盘输入
        str = input("Please enter:");
        print("你输入的内容是: ", str)

