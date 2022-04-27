# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00                   
# Date: 2021/2022                       
# Module: EA arguments.

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    
    # PATHS
    parser.add_argument("--exp_path", type=str, help="Experiment dump path")
   
    # EA params 
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--population_size", type=int, default=5, help="Populatin")
    parser.add_argument("--pm", type=float, default=0.8, help="Probability of mutation of chromozome")
    parser.add_argument("--qm", type=float, default=0.1, help="Probability of mutation of bit in chromozome.")
    parser.add_argument("--pc", type=float, default=0.2, help="Probability of crossover of pair.")
    parser.add_argument("--qc", type=float, default=0.3, help="Probability of crossover of each stage.")

    # Neural Network hyper params
    parser.add_argument("--cnn_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--cnn_nodes", type=str, default="4,5", help="Number of nodes in each stage")

    parser.add_argument("--fc_layers", type=int, default=1, help="Number of fully connected layers.")

    # Training hyper params
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=128)

    # GPU
    parser.add_argument("--gpu", type=int, default=0, help="Compute on gpu")

    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset used to evaluate population.")

    parser.add_argument("--verbose", type=bool, default=False, help="Compute on gpu")
    return parser
