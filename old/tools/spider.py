#! /usr/bin/env python3
#-*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup

server = 'http://www.shicimingju.com'


def get_type_list():
    target = 'http://www.shicimingju.com/shicimark'
    req = requests.get(url = target)
    html = req.text
    div_bf = BeautifulSoup(html)
    div = div_bf.find_all('div', class_ = 'www-shadow-card www-main-container')
    href_bf = BeautifulSoup(str(div[0]))
    href_list = href_bf.find_all('a')
    type_list = [(a.string, server + a.get('href')) for a in href_list]
    return type_list

def get_poetry_list(target):
    req = requests.get(url = target)
    html = req.text
    h3_bf = BeautifulSoup(html)
    h3_list = h3_bf.find_all('h3')
    href_list = [server + BeautifulSoup(str(h3)).find_all('a')[0].get('href') for h3 in h3_list]
    poetry_list = []
    for href in href_list:
        poetry_list.append(get_poetry(href))
    return poetry_list

def get_poetry(target):
    req = requests.get(url = target)
    html = req.text
    div_bf = BeautifulSoup(html)
    div = div_bf.find_all('div', class_ = 'shici-content')
    content = [s.strip() for s in div[0].strings] 
    return ''.join(content)

def gen_poetry_corpus():
    type_list = get_type_list()
    with open('./corpus/poetry.txt', 'w', encoding = 'utf-8') as f:
        for t, href in type_list:
            poetry_list = get_poetry_list(href)
            for poetry in poetry_list:
                f.write(' '.join((t, poetry, '\n')))





if __name__ == '__main__':
    gen_poetry_corpus()