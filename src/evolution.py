# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00
# Date: 2021/2022
# Module: Implementation of EA loop with initialization of population.

from parser import get_parser
from operations import init_population, selection, mutation, crossover, compute_fitness
import json
import random


def ea_loop(params, population):
    dict = {}
    tmp_results = []
    best = {"genotype": "", "acc": 0}

    # experiment values to save
    for chromosome in population:
        tmp_results.append({"genotype": chromosome.genotype, "fitness": chromosome.fitness}) 
    dict['Init'] = tmp_results

    for generation in range(params.generations):
        print("Generation: ", generation)
        
        population = selection(population, params)
        population = mutation(population, params.pm, params.qm)
        population = crossover(population, params.pc, params.qc)

        population = compute_fitness(population, params)
        # exp values to save for each generation
        tmp_results = []
        for chromosome in population:
            tmp_results.append({"genotype": chromosome.genotype, "fitness": chromosome.fitness})
            if chromosome.fitness >= best['acc']:
                best["genotype"] = chromosome.genotype
                best["acc"] = chromosome.fitness
        dict["Gen_"+str(generation)] =  tmp_results
        random.shuffle(population)
        
        if generation % 5 == 0:
            with open(params.exp_path, 'w') as f:
                json.dump(dict, f)
    # save best individual
    dict["best"] = {"genotype": best["genotype"], "fitness": best["acc"]}
        

    with open(params.exp_path, 'w') as f:
        json.dump(dict, f)


if __name__=='__main__':
    parser = get_parser()
    params = parser.parse_args()
    
    if params.gpu != 0:
        from safe_gpu import safe_gpu

        gpu_owner = safe_gpu.GPUOwner()

    population = init_population(params)
    population = compute_fitness(population, params)

    ea_loop(params, population)
