#! /usr/bin/env python
# -*- coding:utf-8 -*-

from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
import re
import sqlite3
import sys

def get_links(url, pref, pattern='.*'):
    print('#', url)
    while True:
        try:
            html = urlopen(url)
        except HTTPError:
            return None
        except URLError as e:
            print('# urlopen error:', e)
        except ConnectionResetError as e:
            print('# Connection reset error', e)
        else:
            break
    bsObj = BeautifulSoup(html, 'html.parser')
    for link in bsObj.findAll('a', href=re.compile('^/' + pref + '/A\d+/A\d+/\d+/$')):
        if 'href' in link.attrs:
            print(link.attrs['href'], link.get_text())


if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)
    if argc == 2:
        pref = argv[1]
        fname = 'linklist_' + pref + '.txt'
        with open(fname, 'r') as f:
            links = f.read().splitlines()
        for link in links:
            get_links(link, pref, '^/' + pref + '/A\d+/A\d+/\d+/$')
    print('# done')
        
        
