# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00
# Date: 2021/2022
# Module: Implementation of operations needed for EA.

from scipy.stats import bernoulli
import numpy as np
import random

from chromosome import Chromosome
from train_eval_ea import train, eval


def init_population(params):
    assert params.cnn_stages == len(params.cnn_nodes.split("_")), "Incorrect number of CNN stages and nodes"
    population = []

    print("Initializing Population")
    for _ in range(params.population_size):
        genotype = create_genotype(params)
        tmp = Chromosome(params.cnn_stages, list(map(int, params.cnn_nodes.split("_"))), genotype, dataset=params.dataset)
        if params.gpu == 1:
            tmp = tmp.cuda()
        population.append(tmp)
    return population 


def create_genotype(params):
    gen = []
    # generate genotype for each stage and concat
    for s in range(params.cnn_stages):
        n_nodes = int(params.cnn_nodes.split("_")[s]) # list of number of nodes for each stage 
        stage_length = int((n_nodes * (n_nodes-1)) / 2 ) 


        bern = list(map(str, np.array(bernoulli.rvs(0.5, size=stage_length))))  # bernoulli probability distribution
        gen_tmp = []

        # generate connection genotype for each stage
        for k in range(n_nodes): 
            gen_tmp.append("".join(bern[:k]))

        gen_tmp = gen_tmp[1:]
        gen.append(gen_tmp)
    print("GENOTYPE: {}, STAGES: {}, NODES: {}".format(gen, params.cnn_stages, params.cnn_nodes))  
    return gen

def mutation(population, pm, qm):
    """ Function that mutates each bit in each chromosome in population with probability p.
    """
    num_mut = 0
    for chromosome in population:
        if random.random() < pm:
            for stage in range(len(chromosome.genotype)):
                # for node composed of CNN layers
                for i, node in enumerate(chromosome.genotype[stage]):
                    tmp_node = ""
                    # for each connection
                    for bit in node:
                        if random.random() < qm:
                            tmp_node = tmp_node + str(1 - int(bit))
                            num_mut += 1
                        else:
                            tmp_node = tmp_node + bit
                    chromosome.genotype[stage][i] = tmp_node
    
    print("Mutating population({}), with pm={}, qm={}. Number of mutations: {}.".format(len(population), pm, qm, num_mut))
    return population


def crossover(population, pc, qc):
    """ Performs crossover of each stage with probability p for two individuals.
    """
    num_cross = 0 
    population_it = iter(population)
    try:
        for chrom1 in population_it:
            chrom2 = next(population_it)
            if random.random() < pc: 
                for stage in range(len(chrom1.genotype)):
                    if random.random() < qc:
                        num_cross += 1
                        chrom1.genotype[stage], chrom2.genotype[stage] = chrom2.genotype[stage], chrom1.genotype[stage]

    except StopIteration:
        # skip if last chromosome if population size is odd number
        pass

    print("Crossover on population({}), with pc={}, qc={}. Number of crowssovers: {}.".format(len(population), pc, qc, num_cross))
    return population


def selection(population, params): 
    """ Performs Russian roulette selection of individuals based on their fitness. Individual
    with lowest fitness is never chosen (assigned fitness 0, if substracting its fitness causing sometimes NaN).
    """
    lowest = 0
    weights = []

    # get fitness values
    for chromosome in population:
        weights.append(chromosome.fitness)
     
    weights = np.array(weights)
    lowest = np.argmin(weights)
    weights[lowest] = 0  # prevents NaN

    # r_n - r_0 than make probability sum to 1
    #weights = (weights - lowest) + 0.00001 
    weights = weights / np.sum(weights)

    if params.verbose: print("Selection ================================")
    if params.verbose: print_p(population)

    # select same number of individuals, weights are norm fitness
    population = np.random.choice(population, size=params.population_size, replace=True, p=weights)
    if params.verbose: print("------------------------------")
    if params.verbose: print_p(population)
    return population

def print_p(population):
    for c in population:
        print(c.genotype, " -> ", c.fitness)

def compute_fitness(population, params):
    for chromosome in population:
        train(params.epochs, chromosome, params.gpu, params.dataset)
        acc = eval(chromosome, params.gpu, params.dataset)
        chromosome.fitness = acc
    return population

def reset_weights(population):
    for chromosome in population:
        chromosome.reset_w()
    return population