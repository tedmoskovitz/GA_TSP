'''
Baseline methods for solving TSP
Ted Moskovitz
2018
'''
import numpy as np  
from utils import path, mutate
from scipy.spatial.distance import cdist
import sys
import time

class random_search:

    def __init__(self, points, findShortest=True, prob=1, run_idx=1):
        '''
        initialize search
        args:
            points: N x 2 ndarray of 2D points
            findShortest: boolean, find shortest or longest path
        '''
        self.points = points.astype(np.float32)
        self.dmatrix = cdist(points, points)
        self.N = points.shape[0]
        self.findShortest = findShortest
        self.best_path = path(np.random.permutation(self.N), self.dmatrix, findShortest=self.findShortest)
        self.best_path_length = self.best_path.d
        self.best_fitness = self.best_path.f
        self.log_dir = './saved/'
        self.prob = prob 
        self.run_idx = run_idx
    
    def run(self, niters, save=True):
        '''
        run a random search
        args: 
            niters: # of iterations 
        '''
        from utils import path
        start = time.time()
        self.fitness_hist = np.zeros(niters)
        for t in range(niters):
            # generate new random ordering
            new_order = np.random.permutation(self.N)
            new_path = path(new_order, self.dmatrix, findShortest=self.findShortest)

            # compare to current best path and replace if better
            if new_path.f > self.best_fitness:
                self.best_path = new_path
                self.best_fitness = new_path.f
                self.best_path_length = new_path.d #self.best_path.calc_length()

            # log current best
            self.fitness_hist[t] = self.best_fitness

            if (t + 1) % 2000 == 0:
                print("\rRandom Search run {}: {}/{} evaluations complete - best length = {}".format(self.run_idx,
                    t + 1, niters, self.best_path_length), end="")
                sys.stdout.flush()
        
        end = time.time()
        print ('\nTotal time: {} min'.format((end-start)/60.0))

        if save:
            s = 'short' if self.findShortest else 'long'
            path = self.log_dir + 'RS_{}evals_TSP{}_{}_run{}_'.format(niters, self.prob, s, self.run_idx)
            np.savetxt(path + 'fitness_hist.csv', self.fitness_hist, delimiter=',')
            np.savetxt(path + 'best_path.csv', self.best_path.order, delimiter=',')
            print ('\nSaved.')



class hill_climber:

    def __init__(self, points, findShortest=True, prob=1, run_idx=1):
        '''
        basic hill climbing algorithm - mutate current best path 
        args:
            points: N x 2 ndarray of 2D points 
            findShortest: boolean, find shortest or longest path 
        '''
        self.points = points
        self.dmatrix = cdist(points, points)
        self.N = points.shape[0]
        self.findShortest = findShortest
        self.best_path = path(np.random.permutation(self.N), self.dmatrix, findShortest=findShortest)
        self.best_path_length = self.best_path.d
        self.best_fitness = self.best_path.f
        self.log_dir = './saved/'
        self.prob = prob
        self.run_idx = run_idx

    def run(self, niters, save=True):
        '''
        run a random mutation hill climber 
        args:
            niters: # of iterations
        '''
        from utils import path
        start = time.time()
        self.fitness_hist = np.zeros(niters)
        for t in range(niters):
             # mutate the current best ordering by swapping a pair of points at random
             new_path = path(mutate(self.best_path.order), self.dmatrix, findShortest=self.findShortest)
             
             # compare to current best path and replace if better
             if new_path.f > self.best_fitness:
                 self.best_path = new_path
                 self.best_fitness = new_path.f
                 self.best_path_length = new_path.d# self.best_path.calc_length()

             # log current best
             self.fitness_hist[t] = self.best_fitness

             if (t + 1) % 2000 == 0:
                 print("\rRMHC run {}: {}/{} evaluations complete - best length = {}".format(self.run_idx,
                    t + 1, niters, self.best_path_length), end="")
                 sys.stdout.flush()
        
        end = time.time()
        print ('\nTotal time: {:.3f} min'.format((end-start)/60.0))

        if save:
            s = 'short' if self.findShortest else 'long'
            path = self.log_dir + 'RMHC_{}evals_TSP{}_{}_run{}_'.format(niters, self.prob, s, self.run_idx)
            np.savetxt(path + 'fitness_hist.csv', self.fitness_hist, delimiter=',')
            np.savetxt(path + 'best_path.csv', self.best_path.order, delimiter=',')
            print ('\nSaved')


        
