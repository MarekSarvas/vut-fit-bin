# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00
# Date: 2021/2022
# Module: 

from parser import get_parser
from operations import init_population, selection, mutation, crossover, compute_fitness
import json



def ea_loop(params, population):
    dict = {}
    tmp_results = []
    for chromosome in population:
        tmp_results.append({"genotype": chromosome.genotype, "fitness": chromosome.fitness.cpu().detach().numpy().item(0)}) 
    dict['Init'] = tmp_results

    best = {"genotype": "", "acc": 0}
    for generation in range(params.generations):
        print("Generation: ", generation)
        
        population = selection(population, params)
        population = mutation(population, params.mut_p)
        population = crossover(population, params.cross_p)

        population = compute_fitness(population, params)
        
        tmp_results = []
        for chromosome in population:
            tmp_results.append({"genotype": chromosome.genotype, "fitness": chromosome.fitness.cpu().detach().numpy().item(0)})
            if chromosome.fitness.cpu().detach().numpy().item(0) >= best['acc']:
                best["genotype"] = chromosome.genotype
                best["acc"] = chromosome.fitness.cpu().detach().numpy().item(0)
        dict["Gen_"+str(generation)] =  tmp_results
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
