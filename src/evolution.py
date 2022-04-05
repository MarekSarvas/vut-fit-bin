from parser import get_parser


def mutate(population, p):
    """ Function that mutates each chromosome in population with probability p.
    """
    print("Mutating population({}), with probability {p}".format(len(population), p))
    pass

def init_population(params):
    pass

def compute_fitness(population, params):
    pass

def ea_loop(params):
    for generation in range(params.generations):
        print("Generation: ", generation)
        population=[]
        mutate(population, params.p)

if __name__=='__main__':
    parser = get_parser()
    params = parser.parse_args()

    population = init_population(params)

    ea_loop(params)
