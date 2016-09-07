#! /usr/bin/env python
# -*- coding:utf-8 -*-

from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
import re
import sys
import time

tabelog = 'http://tabelog.com/'

def get_links(url, pattern='.*'):
    result = set()
    while True:
        try:
            html = urlopen(url)
        except HTTPError:
            return None
        except URLError as e:
            print('# urlopen error:', e)
            time.sleep(10)
        except ConnectionResetError:
            print('# Connection reset error')
            time.sleep(10)
        else:
            break
    bsObj = BeautifulSoup(html, 'html.parser')
    for link in bsObj.findAll('a', href=re.compile(pattern)):
        if 'href' in link.attrs:
            result.add(link.attrs['href'])
#            print(link.attrs['href'], link.get_text())
    return sorted(result)

def get_shoplist(pref):
    result = set()
    url = tabelog + 'sitemap/' + pref + '/'
    areas = get_links(url, '^http://tabelog.com/sitemap/')
    for area in areas:
        print('#', area)
        idxs = get_links(area, '^http://tabelog.com/sitemap/')
        for idx in idxs:
            result.add(idx + '?PG=1')
            print(idx + '?PG=1')
            while True:
                flag = True
                lsts = get_links(idx, 'PG=\d+$')
                if not lsts:
                    break
                lsts = [tabelog + lst[1:] for lst in lsts if lst.startswith('/')]
                for lst in lsts:
                    if lst not in result:
                        flag = False
                        result.add(lst)
                        print(lst)
                if flag:
                    break
    print('#done')

            
if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)

    if argc != 2:
        print('usage: python test1.py pref')
    else:
        get_shoplist(argv[1])

    # python test1.py | tee tmp.txt
    # cat tmp.txt | grep -v "#" | sort > linklist_PREF.txt
