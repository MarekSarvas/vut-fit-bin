# Project: VUT FIT BIN - Neuroevolution
# Author: Marek Sarvas
# Login: xsarva00                   
# Date: 2021/2022                       
# Module: Boxplot and table in latex format of accuracy of each generation on given dataset.

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def eval_box(data, path, dataset, title, verbose):
    fig1, ax = plt.subplots(figsize=(10,6))
    ax.set_title(title + " on " + dataset.upper())

    box = ax.boxplot(data["fitness"], labels=data["generation"], showfliers=False, patch_artist=True)

    plt.setp(box["medians"], color='red')
    plt.setp(box["boxes"], facecolor='lightblue')
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
    plt.ylabel("Classification Accuracy")
    plt.xlabel("Generation")

    plt.tight_layout()
    plt.savefig(path, format="pdf")
    if verbose: plt.show()


def eval_table(data, path, dataset):
    path = path.replace(".pdf", ".tex")
    
    # create dataframe and add generations as column
    df = pd.DataFrame({"min \%": data["min"], "Max \%": data["max"], "Avg \%": data["avg"], "Med \%": data["med"], "Genotype": data["genotype"]})
    df = df.rename_axis('Gen').reset_index()
    
    # style table
    styler = df.style.format({"min \%": '{:.4f}', "Max \%": '{:.4f}', "Avg \%": '{:.4f}', "Med \%": '{:.4f}'})  # set precision
    styler = styler.hide_index()
    styler.applymap_index(lambda v: "font-weight: bold;", axis="columns") # bold first row 

    tex_content = styler.to_latex(convert_css=True)  # convert to latex string
    # add column borders
    re_borders = re.compile(r"begin\{tabular\}\{([^\}]+)\}")
    borders = re_borders.findall(tex_content)[0]
    borders = '|'.join(list(borders))
    tex_content = re_borders.sub("begin{tabular}{|" + borders + "|}", tex_content)

    tex_content = tex_content.replace("\\bfseries", "\hline\n\\bfseries", 1) # add upper border
    tex_content = tex_content.replace("0", "\hline\n 0", 1) # add first row border
    tex_content = tex_content.replace("\\end", "\hline\n\\end", 1) # add last border
    tex_content = tex_content.replace("Med", "\\bfseries Med", 1) # fix last column bold text 
    
    # save as .tex file
    with open(path, "w") as f:
        f.write(tex_content)

def gen2str(genotype):
    result = ""
    for stage in genotype:
        for connection in stage:
            result += str(connection) + "-"
        #result = result[:-1]
        result = " $\\vert$ ".join(result.rsplit("-", 1))
    return result[:-9]

def check_for_connection(stage):
    for node_c in stage:
        if '1' in node_c: return True
    return False


def conv_mult(genotype, dataset):
    d = {"fashion": 28, "mnist": 28, "cifar10": 32}
    stage_channels = 32
    mult = 0

    for i, s in enumerate(genotype):
        if i == 0:
            mult += d[dataset]**2 * 25 * 1
        else:
            mult += (d[dataset]/2*i)**2 * 25 * stage_channels
        if not check_for_connection(s):
            continue
        for v in s:
            for bit in v:
                if bit == '1':
                    mult += (d[dataset]/2*i)**2 * 9 * stage_channels
        stage_channels *= 2
        mult += (d[dataset]/2*i)**2 * 9 * stage_channels
        stage_channels *= 2
    return mult

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
            data = {"generation": np.array([], dtype=object), "fitness": [], "genotype": np.array([], dtype=object), "avg": [], "min": [], "max": [], "med": []}

            with open(json_path) as f:
                json_data = json.load(f)
           
            # for each generation gather data
            for i, item in enumerate(json_data.items()):
                # not dictionary -> skip
                best = ""
                if item[0] == "best":
                    best = gen2str(item[1]["genotype"])
                    continue
                data["generation"] = np.append(data["generation"], str(i))
                tmp = []
                for c in item[1]:
                    tmp.append(c['fitness'])
                 
                # store genotype with highest fitness in current generation
                data["genotype"] = np.append(data["genotype"], gen2str(item[1][np.argmax(tmp)]["genotype"]))
                data["fitness"].append(np.array(tmp))
                data["min"].append(np.min(tmp))
                data["max"].append(np.max(tmp))
                data["avg"].append(np.mean(tmp))
                data["med"].append(np.median(tmp))
            print(plot_path)
            # plot data
            eval_box(data, plot_path, params.dataset, title, params.verbose) 
            eval_table(data, plot_path, params.dataset)
