import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    
    # PATHS
    parser.add_argument("--context_path", type=str, default="./dump/context/train", help="Context images path")
    parser.add_argument("--style_path", type=str, default="./dumped/style/train", help="Style images path")
    parser.add_argument("--exp_path", type=str, help="Experiment dump path")
   
    # EA params 
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--population_size", type=int, default=5, help="Populatin")
    parser.add_argument("--mutation_probability", type=int, default=0.01, help="Probability of mutation of gene.")


    parser.add_argument("--cnn_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--cnn_nodes", type=str, default="4,5", help="Number of nodes in each stage")

    parser.add_argument("--fc_stages", type=int, default=0, help="Number of fully connected layers stages, 0 is turned off.")
    parser.add_argument("--fc_nodes", type=str, default="2,2", help="Number of nodes in each stage")



    # GPU
    parser.add_argument("--gpu", type=int, default=0, help="Compute on gpu")

    return parser
