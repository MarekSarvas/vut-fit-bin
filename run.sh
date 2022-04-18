#!/usr/bin/bash
# Bash template for running dataprep and training the model
. ./path.sh || exit 1;

MAIN_PATH=$(pwd)

cd src/

stage=2
stop_stage=2
if [ $# -eq 2 ]; then
    stage=$1
    stop_stage=$2
fi
dumpdir=dump
verbose=0
baseline=cnn

epochs=3
mutation_p=0.05
crossover_p=0.2
stages=2
nodes="4_5"
dataset="mnist"

exp_id=exp_stages${stages}_nodes${nodes}
tag="exp_mut${mutation_p}_cross${crossover_p}"


EXP_PATH=${MAIN_PATH}/exp/${exp_id}

mkdir -pv ${EXP_PATH}

if [ ${stage} -le 0 ] &&[ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"
    rm -r ${MAIN_PATH}/../data/ 
    mkdir ${MAIN_PATH}/../data/
    python3 download_data.py --data_path ${MAIN_PATH}/data/
fi


if [ ${stage} -le 1 ] &&[ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Baseline Model Training"
    if [ -z ${tag} ]; then
        expname=mnist_${baseline}_epochs_${epochs} 
    else
        expname=mnist_${baseline}_${tag} 
    fi
    mkdir -pv ../exp/${expname}
    python train_eval.py
fi

if [ ${stage} -le 2 ] &&[ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Neuroevolution"
    python3 evolution.py  --generations 2 \
                        --population_size 5 \
                        --epochs ${epochs} \
                        --exp_path ${EXP_PATH}/${tag}.json \
                        --mut_p ${mutation_p} \
                        --cross_p ${crossover_p} \
                        --cnn_stages ${stages} \
                        --cnn_nodes ${nodes} \
                        --gpu 1 \
                        --dataset ${dataset} \
                        --verbose True \
                #        > ${EXP_PATH}/${tag}.log 
fi

