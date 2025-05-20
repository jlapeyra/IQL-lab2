import math
from typing import Sequence
import numpy as np
from utils import windowed, flatten
import random
from tqdm import tqdm
import re
import os
from distribution import Distribution, ConditionalDistribution
from utils import hash
import sys
import gzip

NULL = r'\0'
NEWLINE = r'\n'
LINES_PER_MODEL_FILE = 100_000

class NGram(ConditionalDistribution):
    # Counter allows __add__

    def __init__(self, n:int, path_prefix:str): 
        super().__init__()
        self.n = n
        self.null = NULL
        self.path = path_prefix
        self.__load_mode = False

    def __sliding_window(self, sequence:Sequence):
        # prior     = map(self.func_prior, sequence)
        # posterior = map(self.func_posterior, sequence)
        # prior     = list(chain(head, prior))
        # posterior = list(chain(posterior, tail))
        head, tail = [self.null]*(self.n-1), [self.null]
        sequence = list(sequence)
        prior = head + sequence
        posterior = sequence + tail
        return zip(windowed(prior, self.n-1), posterior)

    def feed(self, sequence:Sequence): # provide some text to train the model
        if self.n == 1:
            pass
        for k1, k2 in self.__sliding_window(sequence):
            assert k2 != '', sequence
            self[k1][k2] += 1
        return self
    
    def __filename(self):
        return f'{self.path}.model'
    
    def __filename_info(self):
        return f'{self.path}.info'
    
    def __filename_part(self, num):
        return f'{self.path}.{num:04}.part'
    
    # Save part
    def __save(self, flat_dict:dict, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for (prior, posterior), count in sorted(flat_dict.items()):
                print(*prior, posterior, count, file=f)

    # Save by parts
    def save(self, min_occurrences=1):
        dict = {k:v for k,v in self.items() if v.total() > min_occurrences}
        flat_dict = flatten(self)
        num_lines = len(flat_dict)
        if self.n == 1 or num_lines < 2*LINES_PER_MODEL_FILE:
            self.__save(flat_dict, self.__filename())
        else:
            self.__num_files = num_lines // LINES_PER_MODEL_FILE
            with open(self.__filename_info(), 'w') as f:
                f.write(str(self.__num_files))
            for i in tqdm(range(self.__num_files), f'Saving {self.n}-gram', leave=False):
                if hash(('\\0', '\\0'))%self.__num_files == i:
                    pass
                self.__save(
                    flatten({k:v for k,v in self.items() if hash(k)%self.__num_files == i}), 
                    filename=self.__filename_part(i)
                )


    # Prepare to load by parts
    def load(self, reverse=False):
        self.__reverse = reverse
        if os.path.exists(path := self.__filename()):
            self.__load(path)
        elif os.path.exists(path_info := self.__filename_info()):
            self.__load_mode = True
            with open(path_info, 'r', encoding='utf-8') as f:
                self.__num_files = int(f.readline().strip())
                for i in range(self.__num_files):
                    assert os.path.exists(self.__filename_part(i))
            self.loaded = np.zeros(self.__num_files)
        else:
            raise ValueError(f'Model not found. At least one of these files should exist: {path} {path_info}')


    # Load part
    def __load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f: #tqdm(f.readlines()):
                try:
                    *keys, count = line.strip('\n').split()
                    if self.__reverse:
                        keys = reversed(keys)
                    *prior, posterior = keys
                    assert len(prior) + 1 == self.n
                    count = int(count)
                except:
                    raise Exception(f'Wrong format: expected {self.n} keys and a number. Got: "{line}"')
                else:
                    self[tuple(prior)][posterior] += count

    def getProbDistrib(self, key):
        assert isinstance(key, tuple)
        if self.__load_mode:
            h = hash(key) % self.__num_files
            if not self.loaded[h]:
                self.loaded[h] = True
                self.__load(self.__filename_part(h))
        return self[key]

    
class Model:  #MetaNGram
    ngrams : list[NGram]

    def __init__(self, n, name):
        self.ngrams = []
        self.name = name
        for i in range(1, n+1):
            self.ngrams.append(NGram(i, f'model/{name}/{i}-gram'))

    def __getitem__(self, key) -> NGram:
        return self.ngrams[key]

    def __iter__(self):
        return iter(self.ngrams)
    
    def __len__(self):
        return super().__len__()

    def generate(self, prompt:str='', do_print=True):
        max_prior = len(self.ngrams) - 1
        INIT = tuple([NULL]*max_prior)
        prompt = tokenize(prompt)
        prior = INIT + tuple(prompt)
        prior = prior[len(prior)-max_prior:]

        words = list(prompt)
        word = ''
        while word != NULL:
            for ngram in reversed(self.ngrams):
                prior_size = ngram.n - 1
                distrib = ngram.getProbDistrib(prior[max_prior-prior_size:])
                if distrib.total() >= random.randint(1, 2):
                    break
            word = distrib.randomChoice()
            prior = prior[1:] + (word,)
            if do_print:
                print(untokenize([word], words), end='')
                sys.stdout.flush()
            words.append(word)
        return untokenize(words)

    def feed(self, string:str):
        for ngram in self.ngrams:
            ngram.feed(tokenize(string))
        return self

    def load(self, reverse=False):
        for ngram in self.ngrams:
            ngram.load(reverse=reverse)
        return self

    def save(self):
        for ngram in self.ngrams:
            ngram.save(min_occurrences=min(1, ngram.n-3))

    def compress(self):
        model_dir = f'model/{self.name}/'
        for filename in tqdm(os.listdir(model_dir), 'compressing '+self.name):
            filepath = os.path.join(model_dir, filename)
            if os.path.isfile(filepath) and not filepath.endswith('.gz'):
                with open(filepath, 'rb') as f_in, gzip.open(filepath + '.gz', 'wb') as f_out:
                    f_out.write(f_in.read())

    def uncompress(self):
        model_dir = f'model/{self.name}/'
        for filename in os.listdir(model_dir):
            filepath = os.path.join(model_dir, filename)
            if os.path.isfile(filepath) and filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f_in, open(filepath[:-3], 'wb') as f_out:
                    f_out.write(f_in.read())

def tokenize(string:str) -> list[str]:
    tokens = re.findall(r"\w[\w'\-·]*\w|\w|\d[\d.,]*\d|\.\.\.|[^\w\s]|\n", string)
    return [t if t!='\n' else NEWLINE for t in tokens]

def untokenize(tokens:list[str], previous_tokens:list[str]=[]):
    prev = (['\n'] + previous_tokens)[-1]
    ret = []
    for token in tokens:
        if token in (NEWLINE, NULL):
            token = '\n'
        if not (
            '\n' in (prev, token)
            or __is_pre_punct(prev)
            or __is_post_punct(token)
        ):
            ret.append(' ')
        ret.append(token)
        prev = token
    return ''.join(ret)


def __is_word(token):
    return bool(re.match(r"[\w'\-·]+", token))

def __is_post_punct(token):
    return token in '.,:;)]}?!' or token == '...'

def __is_pre_punct(token):
    return token in '({[¿¡'

