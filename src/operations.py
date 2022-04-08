# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00
# Date: 2021/2022
# Module: 

from scipy.stats import bernoulli
import numpy as np

from chromosome import Chromosome

def init_population(params):
    assert params.cnn_stages == len(params.cnn_nodes.split(",")), "Incorrect number of CNN stages and nodes"
    population = []
    print("Initializing Population")
    for _ in range(params.population_size):
        genotype = create_genotype(params)
        tmp = Chromosome(params.cnn_stages, list(map(int, params.cnn_nodes.split(","))), genotype) 
        population.append(tmp)
        # print(tmp.genotype, tmp.fitness)
    return population 


def create_genotype(params):
    gen = ""
    # generate genotype for each stage and concat
    for s in range(params.cnn_stages):
        n_nodes = int(params.cnn_nodes.split(",")[s])
        stage_length = int((n_nodes * (n_nodes-1)) / 2 )

        #bern = np.array(bernoulli.rvs(0.5, size=stage_length))
        bern = list(map(str, np.array(bernoulli.rvs(0.5, size=stage_length))))
        
        gen_tmp = ""
        # generate connection genotype for each stage
        for k in range(n_nodes): 
            #gen = gen + str(bern[:connections[k]) + "|"
            gen_tmp = gen_tmp + "".join(bern[:k]) + "-"

        gen_tmp = gen_tmp[1:-1]
        gen = gen + "|" + gen_tmp
        # print("Genotype {}, stage {}".format(gen, s))
    
    # remove excess "|" at the beginning
    gen = gen[1:]
    # print("Init genotype {}".format(gen))
    return gen

def mutation(population, p):
    """ Function that mutates each chromosome in population with probability p.
    """
    print("Mutating population({}), with probability {}".format(len(population), p))
    return population

def selection(population):
    return population

def crossover(population, p):
    print("Crossover on population({}), with probability {}".format(len(population), p))
    return population 
