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
import seaborn as sns


def eval_table(data, path, dataset):
    path = path.replace(".pdf", ".tex")
    number_of_layers(data["genotype"])
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

def number_of_convs_per_stage(genotype):
    nums_per_stage = []
    for s in genotype:
        s_conv = np.zeros(len(s)+1)

        if not check_for_connection(s):  
            nums_per_stage.append(1)
            continue 
        else:
            for v in s:
                for i, bit in enumerate(v):
                    if bit == '1':
                        s_conv[i] = 1 
                        s_conv[i+1] = 1
        
        nums_per_stage.append(np.sum(s_conv, dtype=int)+1) # with first default conv 
    print(genotype, nums_per_stage)
    return nums_per_stage


def MACs_last_gen(data, labels):
    MACs = []
    exp = []
    acc = []
    for i, d in enumerate(data):
        for chromosome in d:
            #MAC_from_genotype([['0', '01'], ['1', '11']])
            MACs.append(MAC_from_genotype(chromosome["genotype"]))
            exp.append(labels[i])
            acc.append(chromosome["fitness"])
    
    df = pd.DataFrame({"MACs": MACs, "Exp": exp, "Acc": acc})
    print(df)
    ax = sns.scatterplot(y="MACs", x="Acc", data=df, hue="Exp")
    ax.set(xscale="log", yscale="log")

    plt.show()

def MAC_from_genotype(genotype):
    convs = number_of_convs_per_stage(genotype)
    macs = 0
    img_size = 28*28
    in_channels = 16
    out_channels = 16
    for i, c in enumerate(convs):
        if(i == 0):
            macs += 5*5*img_size*in_channels
        else:
            macs += 5*5*img_size*(in_channels/2) * out_channels
        for i in range(c):
            macs += 3*3*img_size*in_channels*out_channels

        in_channels *= 2 
        out_channels *= 2 
        img_size /= 2 
    print(genotype, macs)
    return macs

def MACs_best_every_gen(data):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str)
    parser.add_argument("--path2", type=str)
    parser.add_argument("--verbose", action='store_true')
    parser.set_defaults(verbose=False)
    params = parser.parse_args()

    last = True 
            
    # store data for plotting
    data = {"generation": np.array([], dtype=object), "fitness": [], "genotype": np.array([], dtype=object), "avg": [], "min": [], "max": [], "med": []}
    
    with open(params.path1) as f:
        json_data1 = json.load(f)

    with open(params.path2) as f:
        json_data2 = json.load(f)
    if last:
        data = []
        for exp in [json_data1, json_data2]:
            generation = list(exp.keys())[-1]
            data.append(exp[generation])
            MACs_last_gen(data, [params.path1.split("/")[-1].split("_")[1], params.path2.split("/")[-1].split("_")[1]])
    else:
        data = []
        for exp in [json_data1, json_data2]:
            generation = list(exp.keys())[-1]
            data.append(exp[generation])
        MACs_last_gen(data)

    exit()

        
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
        data["# conv layers"]  = item[1][np.argmax(tmp)]["genotype"]
        data["genotype"] = np.append(data["genotype"], gen2str(item[1][np.argmax(tmp)]["genotype"]))
        data["fitness"].append(np.array(tmp))
        data["min"].append(np.min(tmp))
        data["max"].append(np.max(tmp))
        data["avg"].append(np.mean(tmp))
        data["med"].append(np.median(tmp))
    
    # plot data
    eval_table(data, plot_path, params.dataset)
