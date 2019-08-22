#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: email_ops.py
@time: 2019-08-22 17:02
"""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(subject, text, smtpserver='smtp.163.com',
               username='13585581243@163.com',
               password='aEkBI36saG9Kh',  # 授权码
               sender='13585581243@163.com',
               receiver=['13585581243@163.com'],
               msg_from='ian@163.com <ian@163.com>'
               ):
    """

    :param subject: 邮件主题
    :param text: 邮件文本内容
    :param smtpserver:
    :param username:
    :param password:
    :param sender:
    :param receiver:
    :param msg_from:
    :return:
    """
    # 下面的主题，发件人，收件人，日期是显示在邮件页面上的。
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = msg_from
    # 收件人为多个收件人,通过join将列表转换为以;为间隔的字符串
    msg['To'] = ";".join(receiver)

    # 构造文字内容
    text_plain = MIMEText(text, 'plain', 'utf-8')
    msg.attach(text_plain)
    # 发送邮件
    smtp = smtplib.SMTP()
    smtp.connect('smtp.163.com')
    smtp.login(username, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()
