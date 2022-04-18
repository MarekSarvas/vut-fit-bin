# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00
# Date: 2021/2022
# Module: 

from scipy.stats import bernoulli
import numpy as np
import random

from chromosome import Chromosome
from train_eval_cifra import train, eval

def init_population(params):
    assert params.cnn_stages == len(params.cnn_nodes.split("_")), "Incorrect number of CNN stages and nodes"
    population = []
    print("Initializing Population")
    for _ in range(params.population_size):
        genotype = create_genotype(params)
        tmp = Chromosome(params.cnn_stages, list(map(int, params.cnn_nodes.split("_"))), genotype)
        if params.gpu == 1:
            tmp = tmp.cuda()
        population.append(tmp)
    return population 


def create_genotype(params):
    gen = []
    # generate genotype for each stage and concat
    for s in range(params.cnn_stages):
        n_nodes = int(params.cnn_nodes.split("_")[s])
        stage_length = int((n_nodes * (n_nodes-1)) / 2 )

        #bern = np.array(bernoulli.rvs(0.5, size=stage_length))
        bern = list(map(str, np.array(bernoulli.rvs(0.5, size=stage_length))))
        gen_tmp = []

        # generate connection genotype for each stage
        for k in range(n_nodes): 
            #gen = gen + str(bern[:connections[k]) + "|"
            #gen_tmp = gen_tmp + "".join(bern[:k]) + "-"
            gen_tmp.append("".join(bern[:k]))

        gen_tmp = gen_tmp[1:]
        gen.append(gen_tmp)
    print("GENOTYPE: {}, STAGES: {}, NODES: {}".format(gen, params.cnn_stages, params.cnn_nodes))  
    # remove excess "|" at the beginning
    # gen = gen[1:]
    # print("Init genotype {}".format(gen))
    return gen

def mutation(population, p):
    """ Function that mutates each bit in each chromosome in population with probability p.
    """
    num_mut = 0
    for chromosome in population:
        for stage in range(len(chromosome.genotype)):
            # for node composed of CNN layers
            for i, node in enumerate(chromosome.genotype[stage]):
                tmp_node = ""
                # for each connection
                for bit in node:
                    if random.random() < p:
                        tmp_node = tmp_node + str(1 - int(bit))
                        num_mut += 1
                    else:
                        tmp_node = tmp_node + bit
                chromosome.genotype[stage][i] = tmp_node
    
    print("Mutating population({}), with probability {}. Number of mutations: {}.".format(len(population), p, num_mut))
    return population


def crossover(population, p):
    num_cross = 0 
    population_it = iter(population)
    try:
        for chrom1 in population_it:
            chrom2 = next(population_it)

            for stage in range(len(chrom1.genotype)):
                if random.random() < p:
                    num_cross += 1
                    chrom1.genotype[stage], chrom2.genotype[stage] = chrom2.genotype[stage], chrom1.genotype[stage]

    except StopIteration:
        # skip if last chromosome if population size is odd number
        pass

    print("Crossover on population({}), with probability {}. Number of crowssovers: {}.".format(len(population), p, num_cross))
    return population


def selection(population, params): 
    # breakpoint()
    lowest = 0
    weights = []
    # get fitness values
    for chromosome in population:
        weights.append(chromosome.fitness.cpu())
     
    weights = np.array(weights)
    lowest = np.min(weights)
    
    # r_n - r_0 than make probability sum to 1
    weights = (weights - lowest) 
    weights = weights / np.sum(weights)

    if params.verbose: print("Selection ================================")
    if params.verbose: print_p(population)

    population = np.random.choice(population, size=params.population_size, replace=True, p=weights)
    if params.verbose: print("------------------------------")
    if params.verbose: print_p(population)
    return population

def print_p(population):
    for c in population:
        print(c.genotype, " -> ", c.fitness)

def compute_fitness(population, params):
    for chromosome in population:
        train(params.epochs, chromosome, params.gpu)
        acc = eval(chromosome, params.gpu)
        chromosome.fitness = acc
    return population

