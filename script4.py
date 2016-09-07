#! /usr/bin/env python
# -*- coding:utf-8 -*-

from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import time
import sqlite3


prefs = ['aichi','akita','aomori','chiba','ehime',
         'fukui','fukuoka','fukushima','gifu','gunma',
         'hiroshima','hokkaido','hyogo','ibaraki','ishikawa',
         'iwate','kagawa','kagoshima','kanagawa','kochi',
         'kumamoto','kyoto','mie','miyagi','miyazaki',
         'nagano','nagasaki','nara','niigata','oita',
         'okayama','okinawa','osaka','saga','saitama',
         'shiga','shimane','shizuoka','tochigi','tokushima',
         'tokyo','tottori','toyama','wakayama','yamagata',
         'yamaguchi','yamanashi']

def set_allshops():
    with open('shops_all.txt', 'r') as f:
        lines = f.read().splitlines()
    shops = [line.split()[0] for line in lines]
    return shops

allshops = set_allshops()

def get_links(url, pattern='.*'):
    print('#', url)
    global allshops
    result = set()
    while True:
        try:
            html = urlopen(url)
            bsObj = BeautifulSoup(html, 'html.parser')
        except HTTPError:
            return None
        except URLError as e:
            print('# urlopen error:', e)
            time.sleep(10)
        except ConnectionResetError as e:
            print('# connection reset error:', e)
            time.sleep(10)
        else:
            break
#    bsObj = BeautifulSoup(html, 'html.parser')
    for link in bsObj.findAll('a', href=re.compile(pattern)):
        if 'href' in link.attrs:
            u = link.attrs['href']
            if u in result:
                continue
            if u.startswith('https://s.tabelog'):
                continue
            o = urlparse(u)
            if o.path in allshops or o.scheme + '://' + o.netloc + o.path in allshops:
                result.add(o.path)
    return result
            

if __name__ == '__main__':
    # database setting
    conn = sqlite3.connect('./shop_connections.db')
    cur = conn.cursor()
    # initialize table
    # cur.execute("""CREATE TABLE connections(src text, dst text);""")

    # add data
    for idx, src in enumerate(allshops[41389:100000], 41389):
        print('#', idx, end=': ')
        links = get_links('http://tabelog.com' + src)
        if not links:
            continue
        for link in links:
            cur.execute("""INSERT INTO connections(src, dst) VALUES(?, ?)""",
                        (src, link))
        conn.commit()
 
    # show data
    # cur.execute("""SELECT src, dst FROM connections;""")
    # for s, d in cur.fetchall():
    #     print(s, '\t', d)

    conn.close()

        
    

    

