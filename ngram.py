import math
from typing import Sequence
import numpy as np
from utils import windowed
import random
from tqdm import tqdm
import re
import os
from distribution import Distribution, ConditionalDistribution

NULL = '---'
LINES_PER_MODEL_FILE = 10_000

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
    
    def flatten_filter(self, filter): # flatten the nested dicts in one single dict (used for saving)
        ret = {}
        for prior, dist in self.items():
            if filter(prior):
                for posterior, count in dist.items():
                    ret[prior, posterior] = count
        return ret
    
    def flatten(self):
        return self.flatten_filter(lambda _: True)
    
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
    def save(self):
        flat = self.flatten()
        num_lines = len(flat)
        if self.n == 1 or num_lines < 2*LINES_PER_MODEL_FILE:
            self.__save(flat, self.__filename())
        else:
            self.__num_files = num_lines // LINES_PER_MODEL_FILE
            with open(self.__filename_info(), 'w') as f:
                f.write(str(self.__num_files))
            for i in tqdm(range(self.__num_files), f'Saving {self.n}-gram', leave=False):
                self.__save(
                    self.flatten_filter(lambda k: hash(k)%self.__num_files == i), 
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
            ValueError(f'Model not found. At least one of these files should exist: {path} {path_info}')


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

    def __getitem__(self, key):
        assert isinstance(key, tuple)
        if self.__load_mode:
            h = hash(key) % self.__num_files
            if not self.loaded[h]:
                self.loaded[h] = True
                self.__load(self.__filename_part(h))
        return super().__getitem__(key)

    
class Model:  #MetaNGram
    ngrams : list[NGram]

    def __init__(self, n, name):
        self.ngrams = []
        for i in range(1, n+1):
            self.ngrams.append(NGram(i, f'model/{name}/{i}-gram'))

    def __getitem__(self, key) -> NGram:
        return self.ngrams[key]

    def __iter__(self):
        return iter(self.ngrams)
    
    def __len__(self):
        return super().__len__()

    def generate(self, prompt=(), size=None):
        max_prior = len(self.ngrams)
        INIT = tuple([NULL]*max_prior)
        prior = INIT + tuple(prompt)
        prior = prior[len(prior)-max_prior:]
        words = []
        for _ in range(size):
            for ngram in reversed(self.ngrams):
                prior_size = ngram.n - 1
                distrib = ngram[prior[max_prior-prior_size:]]
                if distrib.total() > random.randint(1, 2):
                    break
            word = distrib.randomChoice()
            if word == NULL:
                prior = INIT
            else:
                prior = prior[1:] + (word,)
            words.append(word)
            #print((word if word!='\n' else repr(word)).ljust(16), ngram.n, distrib.total())
        #return words
        print(*words)

    def feed(self, string:str):
        for ngram in self.ngrams:
            ngram.feed(string.split())
        return self

    def load(self, reverse=False):
        for ngram in self.ngrams:
            ngram.load(reverse=reverse)
        return self

    def save(self):
        for ngram in self.ngrams:
            ngram.save()        