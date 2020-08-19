# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:53:23 2020

@author: biejds
"""

import multiprocessing as mp

def kamal_add(a,b):
    return (a+b)


def pool_add():
    with mp.Pool() as pool:
        a = pool.starmap(kamal_add, [(2,3),(4,5),(4,6)])
    print(a)
    return