from collections import Counter, defaultdict
from io import TextIOWrapper
import math
import numpy as np


ALPHA = 0.03

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
            self.__total = sum(self.values())
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
    




        

