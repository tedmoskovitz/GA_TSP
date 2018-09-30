'''
Implementation of a basic genetic algorithm (GA) for solving TSP
Ted Moskovitz
2018
'''
import numpy as np 
from scipy.spatial.distance import cdist
import random
import time 
import sys
from utils import path, mutate, rand_tuple, cross_over

class GA: 

    def __init__(self, points, findShortest=True, prob=1, run_idx=1):
        '''
        a simple genetic algorithm
        args:
            points: N x 2 ndarray of 2D points 
            findShortest: find shortest or longest path through points
            prob: problem #
            run_idx: run #
        '''
        self.N = points.shape[0]
        self.points = points.astype(np.float32)
        self.findShortest = findShortest
        self.dmatrix = cdist(points, points)
        self.prob = prob
        self.run_idx = run_idx

    def run(self, population_size=100, n_gens=1000, p_cross=0.7, p_mut=0.5, roulette=False):
        '''
        run the algorithm for n_gens evaluations
        args:
            population_size: size of population
            n_gens: number of evaluations to run
            p_cross: the crossover probability
            p_mut: the mutation probability 
        '''
        # each row is a chromosome # possibly change population to a set? then need a chromosome/path class with prop calc length, shuffle, fitness
        population = []
        # initialize each chromosome as a random permutation of the points (by index)
        for _ in range(population_size):
        	population.append(path(np.random.permutation(self.N), self.dmatrix, findShortest=self.findShortest))
        
        self.fitness_hist = []
        self.best_path = None

        gen = 0
        self.best_fit = max([x.f for x in population])
        self.best_path_length = -1.0 * self.best_fit if self.findShortest else self.best_fit
        fitness_convergence = []
        total_start = time.time()
        gen_start = time.time()

        while gen < n_gens:
            # population fitness 
            fitness_raw = np.asarray([x.f for x in population])
            # record population convergence
            cnum = -12.56 if self.prob == 1 else -30.0
            fitness_convergence.append(sum(fitness_raw > cnum) / float(population_size))
            # normalize fitness values to use as a valid probability dist
            fitness_norm = fitness_raw / np.sum(fitness_raw)

            gen_best_idx = np.argmax(fitness_raw)
            gen_best_fit = fitness_raw[gen_best_idx]
            gen_best_path = population[gen_best_idx]
            gen_best_length = -1.0 * gen_best_fit if self.findShortest else gen_best_fit
            gen_mean_fit = np.mean(fitness_raw)
            if gen_best_fit > self.best_fit:
                self.best_path = gen_best_path
                self.best_path_length = gen_best_length
                self.best_fit = gen_best_fit

            # repeat until population_size offspring have been created
            new_population = [gen_best_path] # elitism
            while (len(new_population) < population_size):
                if roulette:
                    # roulette wheel selection: 
                    parent1_idx, parent2_idx = np.random.choice(np.arange(population_size),
                     	size=2, replace=True, p=fitness_norm)
                    parent1, parent2 = population[parent1_idx], population[parent2_idx]
                else:
                    # tournament selection:
                    k = 24
                    tourn1 = np.random.choice(np.arange(population_size), size=k, replace=False)
                    tourn1 = [population[i] for i in tourn1]
                    parent1 =  max(tourn1, key=lambda p: p.f)
                    tourn2 = np.random.choice(np.arange(population_size), size=k, replace=False)
                    tourn2 = [population[i] for i in tourn2]
                    parent2 = max(tourn2, key=lambda p: p.f)

                # crossover parents with probability p_cross
                do_cross = random.random()
                if do_cross < p_cross:
                    child1o, child2o = cross_over(parent1.order, parent2.order)
                else:
                    child1o, child2o = (parent1.order, parent2.order)

                # mutate offspring 
                r1, r2 = np.random.rand(2)
                if r1 < p_mut: 
                    child1o = mutate(child1o)
                if r2 < p_mut:
                    child2o = mutate(child2o)

                # add offspring to new population
                child1 = path(child1o, self.dmatrix, findShortest=self.findShortest)
                child2 = path(child1o, self.dmatrix, findShortest=self.findShortest)
                gen += 2
                self.fitness_hist += [gen_best_fit, gen_best_fit]

                best_p = parent2 if parent2.f > parent1.f else parent1
                best_c = child1 if child1.f > child2.f else child2

                new_population.append(best_c)
                new_population.append(best_p)


                # break off
                if len(new_population) > population_size:
                    new_population = new_population[:population_size]
             
            if gen % 100 == 0:
                gen_end = time.time()
                print ('\rGA run {} evaluation {:d}/{:d}: best fitness = {:.4f}, mean fitness = {:.9f},'.format(self.run_idx,gen,
                    n_gens, gen_best_fit, gen_mean_fit)
                    +  ' overall best path = {:.3f}; {:.3f} secs/gen'.format(self.best_path_length,
                    	(gen_end - gen_start)/100.0), end="")
                sys.stdout.flush()
                gen_start = time.time()

            population = new_population

      
        total_end = time.time()
        
        print ('\nSaving results...')
        
        s = 'short' if self.findShortest else 'long'
        prefix = 'saved/GA2_{}evals_TSP{}_{}_run{}'
        
        np.savetxt(prefix.format(n_gens, self.prob, s, self.run_idx) + '_fitness_hist.csv',
            np.asarray(self.fitness_hist), delimiter=',')
        np.savetxt(prefix.format(n_gens, self.prob, s, self.run_idx) + '_best_path.csv',
            np.asarray(self.best_path.order), delimiter=',')
        np.savetxt(prefix.format(n_gens, self.prob, s, self.run_idx) + '_convergence.csv',
            np.asarray(fitness_convergence), delimiter=',')
        print ('Done. \nTotal training time: {:.3f} min'.format((total_end - total_start)/60.0))
        


     




