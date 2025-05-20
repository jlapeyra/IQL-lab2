from abc import abstractmethod
from html.parser import HTMLParser
from time import time
import requests
from utils import overlap
import re
import os
import glob

LANG = 'en'

WIKIPEDIA_DOMAIN = f'https://{LANG}.wikipedia.org'

class TextHTMLParser(HTMLParser):
    __stack:list
    __count_keys:dict[str,int]
    text:str

    def __init__(self) -> None:
        self.text = ''
        self.__stack = []
        self.__count_keys = {
            'include 1' : 0,
            'include 2' : 0,
            'exclude' : 0,
            'caps': 0
        }
        self.__pending_marker = None
        super().__init__()

    def __marker(self, tag):
        if tag == 'li':
            return ' - '
        elif tag[0] == 'h':
            return '#'*int(tag[1:]) + ' '
        else:
            return None


    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        keys = []
        if 'mw-parser-output' in classes or 'firstHeading' in classes: 
            keys.append('include 1')
        if tag in ('p', 'li', 'h1', 'h2', 'h3', 'h4'): #'figcaption', 
            keys.append('include 2')
            self.__pending_marker = self.__marker(tag)
        if (tag in ('sup', 'script', 'style', 'math') or
            overlap(classes, ('mw-editsection', 'reflist', 'citation', 'interProject'))
        ):
            keys.append('exclude')
        if 'font-variant:small-caps' in attrs.get('style', ''):
            keys.append('caps')
        for key in keys:
            self.__count_keys[key] += 1
        self.__stack.append((tag, keys))
        

    def handle_endtag(self, tag):
        stack_tag, keys = None, None
        while tag != stack_tag:
            stack_tag, keys = self.__stack.pop()
            for key in keys:
                self.__count_keys[key] -= 1
                if key == 'include 2':
                    if self.text[-1:] != '\n':
                        self.text += '\n'
    
    def handle_data(self, data):
        if (self.__count_keys['include 1'] and
            self.__count_keys['include 2'] and 
            not self.__count_keys['exclude']
        ):
            if self.__count_keys['caps']:
                data = data.upper()
            if self.__pending_marker:
                self.text += self.__pending_marker
                self.__pending_marker = None
            self.text += data
            #print(self.__count_keys)

class LinksHTMLParser(HTMLParser):
    links:list[str]

    def __init__(self):
        self.links = []
        super().__init__()
    
    def handle_starttag(self, tag, attrs):
        if 'a' == tag and (href := dict(attrs).get('href')) \
        and href.startswith('/wiki/') and isValidPage(href):
            self.links.append(href.removeprefix('/wiki/').split('#')[0])


def getPlainText(html:str):
    p = TextHTMLParser()
    p.feed(html)
    return p.text
    
def getLinks(html:str):
    p = LinksHTMLParser()
    p.feed(html)
    return p.links

def downloadHTML(url):
    response = requests.get(url)
    return response.content.decode(response.encoding)

def isValidPage(link:str):
    return not ':' in link

def saveLinks(links:list):
    with open('wiki-links.txt', 'a', encoding='utf-8') as f:
        for link in links:
            f.write(link + '\n')
    
def saveVisitedLink(link:str):
    with open('wiki-links-visited.txt', 'a', encoding='utf-8') as f:
        f.write(link + '\n')
    
def loadLinks():
    with open('wiki-links.txt', 'r', encoding='utf-8') as f:
        return [line.strip('\n') for line in f]
    
def loadVisitedLinks():
    with open('wiki-links-visited.txt', 'r', encoding='utf-8') as f:
        return [line.strip('\n') for line in f]


def basename(fn:str) -> str:
    return os.path.splitext(os.path.basename(fn))[0]

if __name__ == '__main__':
    def utf8char(codes:str):
        codes = codes.split('%')[1:]  # Split and remove the first empty element
        bytes_seq = bytes(int(code, 16) for code in codes)  # Convert hex codes to bytes
        return bytes_seq.decode('utf-8')  # Decode bytes to UTF-8 string

    for fn in glob.glob('data/{LANG}/*.txt'):
        # title = basename(fn)
        # title = re.sub(r'(%[0-9A-Fa-f]{2})+', lambda match: utf8char(match.group()), title)
        # title = title.replace('_', ' ')
        with open(fn, 'r', encoding='utf-8') as f:
            text = ''.join(('\n' if not line.startswith(' - ') else '') + line for line in f if not (set(line) <= set(' -#\n\t')))
        # text = f'# {title}\n{text}'
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(text)

if __name__ == '__main__':
    import os, glob

    os.makedirs(f'data/wikipedia/{LANG}/', exist_ok=True)
    SAVED_PAGES = set(map(basename, glob.glob(f'data/{LANG}/*.txt')))
    VISITED_PAGES = set(loadVisitedLinks())
    STACK_LINKS = loadLinks() or [
        'Anasayfa'
        #r'Viquip%C3%A8dia:Llista_d%27articles_que_totes_les_lleng%C3%BCes_haurien_de_tenir',
        #r'Viquip%C3%A8dia:Els_100_fonamentals'
    ]


    num_bytes = sum(os.path.getsize(fn) for fn in glob.glob(f'data/{LANG}/*.txt'))
    while STACK_LINKS:
        page = STACK_LINKS.pop(0)
        url = WIKIPEDIA_DOMAIN + '/wiki/' + page
        page = page.replace('/', '%-')
        if page not in VISITED_PAGES:
            VISITED_PAGES.add(page)
            html = downloadHTML(url)
            links = getLinks(html)
            links = [l for l in links if l not in VISITED_PAGES]
            saveLinks(links)
            STACK_LINKS.extend(links)
            if page not in SAVED_PAGES and isValidPage(page):
                SAVED_PAGES.add(page)
                with open(f'data/{LANG}/{page}.txt', 'w', encoding='utf-8') as f:
                    f.write(getPlainText(html))
                num_bytes += os.path.getsize(f'data/{LANG}/{page}.txt')
            saveVisitedLink(page)
            print(len(SAVED_PAGES), len(VISITED_PAGES), len(STACK_LINKS), num_bytes, page)



    