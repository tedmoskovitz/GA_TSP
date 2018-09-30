'''
Useful functions
Ted Moskovitz
2018
'''
import numpy as np 
import matplotlib.pyplot as plt
from numba import jit, jitclass 
from numba import int32, float32, boolean
from scipy.spatial.distance import cdist
import random
import time

@jit(nopython=True, parallel=True)
def randint(lo, hi):
    '''
    generate a random integer in range [lo, hi)
    '''
    int_range = hi - lo 
    return round(lo + random.random() * (int_range-1))

@jit(nopython=True, parallel=True)
def rand_tuple(lo, hi):
    '''
    generate an ordered tuple of random numbers within [lo, hi)
    '''
    idx1 = randint(lo,hi)
    idx2 = randint(idx1,hi)
    return idx1, idx2
    
@jit(nopython=True, parallel=True)
def mutate(order):
    '''
    mutate order by reversing the order of a random contiguous subset
    args:
        order: 1D ordering of point indexes
    returns:
        op: mutated order
    '''
    N = len(order)
    order = list(order)
    idx1 = randint(0, N)
    idx2 = randint(0, N)
    lo, hi = sorted([idx1, idx2])
    op = order[:lo+1] + order[hi:lo:-1] + order[hi+1:]
    return np.array(op)

@jit(nopython=True, parallel=True)
def fitness(order, dmatrix, findShortest=True):
    '''
    fitness function
    args:
        order: ordering of N points in path
        dmatrix: N x N  ndarrary of pairwise distances between points
    returns:
        dist: path distance
        f: path fitness 
    '''
    dist = 0.0
    N = len(order)
    for i in range(1,N):
        dist += dmatrix[order[i],order[i-1]]
    f = -1.0 * dist if findShortest else dist
    return dist, f

@jit(nopython=True, parallel=True)
def cross_over(p1, p2):
    '''
    simple crossover operator 
    args:
        p1 & p2: parent orderings
    returns: 
        c1 & c2: child orderings
    '''

    N = len(p1)
    p1, p2 = list(p1), list(p2)

    idx1, idx2 = rand_tuple(0,N)
    p1_oL = p1[:idx1]
    p1_oR = p1[idx2:]
    p2_s = []
    
    count = 0
    i = 0
    sz = len(p1[idx1:idx2])

    while count < sz:
        p2i = p2[i]
        if p2i not in p1_oL and p2i not in p1_oR:
            p2_s.append(p2i)
            count += 1
        i += 1 

    p2_oL = p2[:idx1]
    p2_oR = p2[idx2:]
    p1_s = []
    
    count = 0
    i = 0
    while count < sz:
        p1i = p1[i]
        if p1i not in p2_oL and p1i not in p2_oR:
            p1_s.append(p1i)
            count += 1
        i += 1

    c1 = p1_oL + p2_s + p1_oR
    c2 = p2_oL + p1_s + p2_oR
    
    return np.array(c1), np.array(c2) 


spec = [
        ('order', int32[:]),
        ('findShortest', boolean),
        ('N', int32),
        ('f', float32),
        ('d', float32)
]

@jitclass(spec)
class path:

    def __init__(self, order, dmatrix, findShortest=True):
        '''
        initialize a path object
        args:
            order: the ordering (by index) of the points in the path
            dmatrix: N x N ndarray of pairwise distances between points on path
        '''
        self.order = order.astype(np.int32)
        self.N = len(order)
        self.findShortest = findShortest
        self.d, self.f = self.fitness(dmatrix)
    
    def fitness(self, dmatrix):
        '''
        fitness function
        '''
        dist = 0.0
        for i in range(1,self.N):
            dist += dmatrix[self.order[i],self.order[i-1]]
        f = -1.0 * dist if self.findShortest else dist
        return dist, f



def main():
    '''
    for testing performance
    '''
    N = 10000000
    pts = np.random.randn(N,2)
    start = time.time()
    print (get_path_len(pts))
    end = time.time()
    print ('Time: {} sec'.format(end-start))

    p = path(np.arange(N), pts, findShortest=True)
    start = time.time()
    print ('Fitness: {}'.format(p.fitness()))
    end = time.time()
    print ('Time: {} sec'.format(end-start))

if __name__=='__main__':
	main()



