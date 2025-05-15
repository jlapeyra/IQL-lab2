from collections import Counter, defaultdict
from io import TextIOWrapper
import math
from typing import Sequence, TypeVar, Callable
import numpy as np
from utils import id, windowed, key_list
from itertools import chain, combinations
from abc import ABC, abstractmethod
import random
from tqdm import tqdm

ALPHA = 0.01
NULL = '---'

class Distribution(Counter):
    # Counter allows __add__

    def add(self, item):
        self[item] += 1
    
    def probability(self, key, alpha:float=ALPHA, num_keys:int=None):
        return (self[key] + alpha)/(self.total() + alpha*(num_keys or len(self.keys())))
    
    def logProbability(self, key, alpha:float=ALPHA, num_keys:int=None):
        return math.log(self.probability(key, alpha, num_keys), 2)

    def probabilityDistribution(self, alpha:float=ALPHA, num_keys:int=None):
        return Distribution({key: self.probability(key, alpha, num_keys) for key in self.keys()})
    
    def logProbabilityDistribution(self, alpha:float=ALPHA, num_keys:int=None):
        return Distribution({key: self.logProbability(key, alpha, num_keys) for key in self.keys()})
    
    prob_array:np.ndarray = None
    __choices = None

    def randomChoice(self, precompute=10):
        assert not self.empty()
        if self.__choices:
            return self.__choices.pop()
        if self.prob_array is None:
            self.prob_array = np.array([self.probability(k) for k in self.keys()])
        assert precompute >= 1
        self.__choices = list(np.random.choice(list(self.keys()), size=precompute, p=self.prob_array))
        if '' in self.__choices:
            pass
        return self.__choices.pop()

    
    def empty(self):
        return len(self) == 0
    
    __total = None
    def total(self):
        if self.__total is None:
            self.__total = super().total()
        return self.__total




class ConditionalDistribution(defaultdict[object, Distribution]):
    def __init__(self) -> None:
        super().__init__(Distribution)

    def add(self, key1, key2, num=1):
        self[key1][key2] += num

    def __add__(self, other:'ConditionalDistribution'):
        if isinstance(other, ConditionalDistribution):
            for key in other.keys():
                self[key] += other[key]

    def save(self, file:TextIOWrapper):
        for keys, count in sorted(self.items()):
            print(*keys, count, file=file)
                
    def load(self, file:TextIOWrapper):
        for line in file.readlines():
            try:
                key1, key2, count = line.strip('\n').split()
                count = int(count)
            except:
                raise Exception(f'Wrong format: expected `key1 key2 num`. Got: "{line}"')
            else:
                self[key1][key2] = count
        return self
    

class NGram(ConditionalDistribution):
    # Counter allows __add__

    def __init__(self, n:int, null=NULL): 
        #, func_prior:Callable=id, func_posterior:Callable=id, func:Callable=None):
        self.n = n
        self.null = null
        # if func is not None:
        #     self.func_posterior = func
        #     self.func_prior = func
        # else:
        #     self.func_prior = func_prior
        #     self.func_posterior = func_posterior
        super().__init__()

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
    
    def flat(self): # flatten the nested dicts in one single dict (used for saving)
        ret = {}
        for prior, dist in self.items():
            for posterior, count in dist.items():
                ret[prior, posterior] = count
        return ret
    
    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for (prior, posterior), count in sorted(self.flat().items()):
                print(*prior, posterior, count, file=f)
                
    def load(self, filename, reverse=False):
        if self.n == 1:
            pass
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                try:
                    *keys, count = line.strip('\n').split()
                    if reverse:
                        keys = reversed(keys)
                    *prior, posterior = keys
                    assert len(prior) + 1 == self.n
                    count = int(count)
                except:
                    raise Exception(f'Wrong format: expected {self.n} keys and a number. Got: "{line}"')
                else:
                    #prior     = tuple(map(self.func_prior, prior))
                    #posterior = self.func_posterior(posterior)
                    self[tuple(prior)][posterior] += count
        return self
    

class MetaNGram(list[NGram]):
    ngrams : list[NGram]

    def __init__(self, n):
        self.ngrams = []
        for i in range(1, n+1):
            self.ngrams.append(NGram(i))

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
                if distrib.total() > random.randint(1, 1):
                    break
            word = distrib.randomChoice()
            if word == NULL:
                prior = INIT
            else:
                prior = prior[1:] + (word,)
            words.append(word)
            #print((word if word!='\n' else repr(word)).ljust(16), ngram.n, distrib.total())
        print(*words)

    def feed(self, sequence: Sequence):
        for ngram in self.ngrams:
            ngram.feed(sequence)
        return self

    def load(self, filename_pattern:str, reverse=False):
        for ngram in self.ngrams:
            ngram.load(filename_pattern.format(n=ngram.n), reverse=reverse)
        return self

    def save(self, filename_pattern:str):
        for ngram in self.ngrams:
            ngram.save(filename_pattern.format(n=ngram.n))
        



if __name__ == '__main__':
    import glob
    data_fn = rf'C:\Users\Usuari\Desktop\MIRI\Q2\IQL\IQL-lab1\europarl\data-v7\raw\europarl-v7.ca-en.ca'

    # model = MetaNGram(n=5)
    # for fn in glob.glob(rf"C:\Users\Usuari\Desktop\Documents\prolliure\hamilton\lletres\*"):
    # with open(data_fn, 'r', encoding='utf-8') as f:
    #     for line in tqdm(f.readlines()[:200_000], 'train'):
    #         model.feed(line.split())
    #     model.save("data/europarl.ca.{n}-gram.model")

    model = MetaNGram(n=4)
    model.load("data/hamilton.en.{n}-gram.model")
    while True:
        prompt = input('Prompt: ')
        prompt = prompt.split()
        model.generate(prompt, 30)

    model.generate(100)


        

