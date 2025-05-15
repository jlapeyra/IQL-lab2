from typing import Callable, Collection, TypeVar, Iterable, Generator
from collections import defaultdict
import numpy as np
import more_itertools
import time
from contextlib import contextmanager

T = TypeVar('T')
K = TypeVar('K')

def group(input:list[T], group_by:Callable[[T], K], container=list) -> dict[K,list[T]]:
    ret = defaultdict(container)
    if container == list: add = list.append
    elif container == set: add = set.add
    else: raise ValueError(f'Container type {container} not supported')
    
    for item in input:
        add(ret[group_by(item)], item)
    return ret

def overlap(set1:Iterable, set2:Iterable):
    return not set(set1).isdisjoint(set2)

def splitList(it:Iterable, sep=None, seps=None, sep_func=None, 
              not_empty=False, keep_sep=True) -> Iterable[list]:
    assert (sep is not None) + (seps is not None) + (sep_func is not None) == 1
    if not sep_func:
        if sep!=None: sep_func = lambda x: x == sep
        else:         sep_func = lambda x: x in seps

    if not_empty:
        valid_part = lambda x: x
    else:
        valid_part = lambda _: True
    
    part = []
    for i, elem in enumerate(it):
        is_sep = sep_func(elem)
        if not is_sep or keep_sep:
            part.append(elem)
        if is_sep:
            if valid_part(part):
                yield part
        
    if valid_part(part):
        yield part

def splitStrip(string:str, sep:str) -> list[str]:
    return list(map, str.strip, string.split(sep))

def idx_map(list:list):
    return {x:i for i, x in enumerate(list)}

def key_list(dict:dict, key_idx:int):
    return sorted(set(map(lambda k: k[key_idx], dict)))

def id(x):
    return x

def windowed(sequence, n):
    if n > 0:
        return more_itertools.windowed(sequence, n)
    else:
        return [() for _ in sequence]

__TIMES = defaultdict(float)

@contextmanager
def timer(name, print_now=False):
    start_time = time.process_time()  # Record start time
    try:
        yield
    finally:
        end_time = time.process_time()  # Record end time
        elapsed_time = end_time - start_time
        __TIMES[name] += elapsed_time
        if print_now:
            print(f"[{name}] Elapsed time: {elapsed_time:.6f} seconds")

def print_times():
    for name, elapsed_time in __TIMES.items():
        print(f"[{name}] Elapsed time: {elapsed_time:.6f} seconds")
