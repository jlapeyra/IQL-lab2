from collections import Counter, defaultdict
from io import TextIOWrapper
import math
from typing import Sequence, TypeVar, Callable
import numpy as np
from utils import id, windowed, key_list
from itertools import chain, combinations
from abc import ABC, abstractmethod


ALPHA = 0.1

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

    def __init__(self, n:int, default=None, func_prior:Callable=id, func_posterior:Callable=id, func:Callable=None):
        self.n = n
        self.default = default
        if func is not None:
            self.func_posterior = func
            self.func_prior = func
        else:
            self.func_prior = func_prior
            self.func_posterior = func_posterior
        super().__init__()

    def __sliding_window(self, sequence:Sequence):
        head, tail = [self.default]*(self.n-1), [self.default]
        prior     = map(self.func_prior, sequence)
        posterior = map(self.func_posterior, sequence)
        prior     = chain(head, prior)
        posterior = chain(posterior, tail)
        return zip(windowed(prior, self.n-1), posterior)

    def feed(self, sequence:Sequence):
        for k1, k2 in self.__sliding_window(sequence):
            self[k1][k2]
        return self
    
    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for (prior, posterior), count in sorted(self.items()):
                print(*prior, posterior, count, file=f)
                
    def load(self, filename, reverse=False):
        with open(filename, 'w', encoding='utf-8') as f:
            for line in f:
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
                    prior     = tuple(map(self.func_prior, prior))
                    posterior = self.func_posterior(posterior)
                    self[prior, posterior] += count
        return self
    
    def generate():
        

    

    

class CoOccurrences(ConditionalDistribution):
    '''
    Co-occurrences of pairs of elements in windows of size n
    '''
    def __init__(self, n:int, func:Callable=id) -> None:
        assert n >= 2
        self.n = n
        self.func = func
        super().__init__()

    def __sliding_window(self, sequence:Sequence, n):
        return windowed(map(self.func, sequence), n)
    
    def feed(self, sequence:Sequence):
        for elems in self.__sliding_window(sequence, self.n):
            for x, y in combinations(elems, 2):
                self[x][y] += 1
                self[y][x] += 1
        return self

    def save(self, filename):
        saved_keys = set()
        with open(filename, 'w', encoding='utf-8') as f:
            for (key1, key2), count in sorted(self.items()):
                if (key2, key1) not in saved_keys:
                    print(key1, key2, count, file=f)
                    saved_keys.add((key1, key2))

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    key1, key2, count = line.strip('\n').split()
                    count = int(count)
                except:
                    raise Exception(f'Wrong format: expected `key1 key2 num`. Got: "{line}"')
                else:
                    self[key1, key2] += count
        return self
    

if __name__ == "__main__":
    print(CoOccurrences(3).feed('abc'))

