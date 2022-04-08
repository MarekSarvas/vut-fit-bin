# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00
# Date: 2021/2022
# Module: 

from parser import get_parser
from operations import init_population, selection, mutation, crossover 


def compute_fitness(population, params):
    print("Computing new fitness")
    pass

def ea_loop(params, population):
    for generation in range(params.generations):
        print("Generation: ", generation)
        population = selection(population)
        population = mutation(population, params.mut_p)
        population = crossover(population, params.cross_p)
        compute_fitness(population, params)
        
        best = {"genotype": "", "acc": 0}
        for chromosome in population:
            if chromosome.fitness >= best['acc']:
                best["genotype"] = chromosome.genotype
                best["acc"] = chromosome.fitness
        print("Genotype -> {}    Accuracy -> {}".format(best["genotype"], best["acc"]))
        print("========================")
        

    best = {"genotype": "", "acc": 0}

    print("Final population")
    for chromosome in population:
        if chromosome.fitness >= best['acc']:
            best["genotype"] = chromosome.genotype
            best["acc"] = chromosome.fitness
        print("Genotype -> {}    Accuracy -> {}".format(chromosome.genotype, chromosome.fitness))
    print("========================")
    print("Genotype -> {}    Accuracy -> {}".format(best["genotype"], best["acc"]))


if __name__=='__main__':
    parser = get_parser()
    params = parser.parse_args()

    population = init_population(params)
    compute_fitness(population, params)
    print("")

    ea_loop(params, population)
