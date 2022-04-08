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
    parser.add_argument("--mut_p", type=float, default=0.05, help="Probability of mutation of gene.")
    parser.add_argument("--cross_p", type=float, default=0.05, help="Probability of mutation of gene.")


    parser.add_argument("--cnn_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--cnn_nodes", type=str, default="4,5", help="Number of nodes in each stage")

    parser.add_argument("--fc_layers", type=int, default=1, help="Number of fully connected layers.")



    # GPU
    parser.add_argument("--gpu", type=int, default=0, help="Compute on gpu")

    return parser
