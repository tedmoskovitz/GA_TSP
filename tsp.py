'''
Solving the Traveling Salesman Problem with Genetic Algorithms
Ted Moskovitz
2018
'''
import numpy as np 
import sys
import matplotlib.pyplot as plt 
import argparse 
from baselines import random_search, hill_climber
from utils import path 
from GA import GA

def main(file_name, method_name, n_gens, findShortest, run_idx, restore_chkpt):
    # read in the data
    points = np.genfromtxt(file_name, delimiter=',')
    prob = int(file_name[3])
    N = points.shape[0]

    # method list
    baselines = ['random search', 'hill climber']
    name2method = {'random search': random_search,
                   'hill climber': hill_climber,
                   'GA': GA}


    if method_name in baselines:
        solver = name2method[method_name](points, findShortest=findShortest, prob=prob, run_idx=run_idx)
        solver.run(n_gens)
    elif method_name == 'GA':
        solver = GA(points, findShortest=findShortest, prob=prob, run_idx=run_idx)
        solver.run(population_size=100, n_gens=n_gens,
        	p_cross=0.3, p_mut=0.9, restore_from_chkpt=restore_chkpt) 
    else:
        raise NotImplementedError('unknown method: options are \'random search,\' \'hill climber,\' \'GA\'')


    # plot path
    path_pts = points[solver.best_path.order.astype(int),:]
    plt.plot(path_pts[:,0], path_pts[:,1],
        label='{} distance: {}'.format(method_name, solver.best_path_length),
        color='C1',  zorder=0, linewidth=0.5)
    plt.scatter(points[:,0], points[:,1], zorder=5)
    plt.legend(loc='upper right')
    plt.show()

    # visualize fitness
    plt.plot(solver.fitness_hist, label=method_name)
    plt.legend()
    plt.xlabel('evaluations')
    plt.ylabel('fitness')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def str2bool(s): 
        if s == 'False': return False;
        elif s == 'True': return True;
        else: raise ValueError('Improper boolean argument.');

    parser.add_argument('--file_name', nargs='?', const='TSP1.txt', type=str)
    parser.add_argument('--method', nargs='?', const='GA', type=str)
    parser.add_argument('--n_gens', nargs='?', const=10000, type=int)
    parser.add_argument('--find_shortest', nargs='?', const=True, type=str2bool)
    parser.add_argument('--run_idx', nargs='?', const=1, type=int)
    args = parser.parse_args()

    main(args.file_name, args.method, args.n_gens, args.find_shortest, args.run_idx, args.restore_chkpt)