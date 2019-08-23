#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: html_ops.py
@time: 2019-08-23 15:06
"""
from pyquery import PyQuery as pq


def create_html(title):
    html = f"""
    <html>  
      <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf8" />
        <title>{title}</title>
      </head>  
      <body>  
      </body>  
    </html> 
    """
    d = pq(html)
    return d


def create_table(thlist, content):
    table = pq('<table border=1 width="80%" align="center"><thead></thead><tbody></tbody></table>')
    tr = '<tr>'
    for th in thlist:
        tr += f'<th>{th}</th>'
    tr += '</tr>'
    table('thead').append(tr)
    for c in content:
        tr = '<tr>'
        for td in c:
            tr += f'<td>{td}</td>'
        tr += '</tr>'
        table('tbody').append(tr)
    return table


def create_img(img_path):
    import base64
    with open(img_path, mode='rb') as f:
        base64_data = base64.b64encode(f.read()).decode()
    from PIL import Image
    im = Image.open(img_path)
    img = pq(f'<img src="data:image/png;base64,{base64_data}" alt="Base64 encoded image" width="{im.size[0]}" height="{im.size[1]}"/>')
    return img
