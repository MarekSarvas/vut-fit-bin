# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00                   
# Date: 2021/2022                       
# Module: Boxplot of accuracy of each generation on given dataset.

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os

def eval_box(data, path, dataset, title, verbose):
    fig1, ax = plt.subplots(figsize=(10,6))
    ax.set_title(title + " on " + dataset.upper())

    box = ax.boxplot(data["fitness"], labels=data["generation"], showfliers=False, patch_artist=True)

    plt.setp(box["medians"], color='red')
    plt.setp(box["boxes"], facecolor='lightblue')
    ax.yaxis.grid(True)
    plt.ylabel("Classification Accuracy")
    plt.xlabel("Generation")

    plt.tight_layout()
    plt.savefig(path, format="pdf")
    if verbose: plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--exp_root", type=str)
    parser.add_argument("--verbose", action='store_true')
    parser.set_defaults(verbose=False)
    params = parser.parse_args()
    
    for path, subdirs, files in os.walk(params.exp_root):
        for name in files:

            json_path = os.path.join(path, name)
            plot_path = "_".join(os.path.join(path, name).replace("/exp/", "/plots/").rsplit("/", 1)).replace(".json", ".pdf")

            title = (path.rsplit("/", 1)[-1]+name).replace(".json", "").replace("exp_", "")
            
            # store data for plotting
            data = {"generation": np.array([], dtype=object), "fitness": [], "genotype": np.array([], dtype=object)}

            with open(json_path) as f:
                json_data = json.load(f)
           
            # for each generation gather data
            for i, item in enumerate(json_data.items()):
                # not dictionary -> skip
                if item[0] == "best":
                    continue
                data["generation"] = np.append(data["generation"], str(i))
                tmp = []
                for c in item[1]:
                    tmp.append(c['fitness'])

                # store genotype with highest fitness in current generation
                data["genotype"] = np.append(data["genotype"], item[1][np.argmax(tmp)]["genotype"])
                data["fitness"].append(np.array(tmp))

            # plot data
            eval_box(data, plot_path, params.dataset, title, params.verbose) 
