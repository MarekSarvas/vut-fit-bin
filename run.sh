#!/usr/bin/bash
# Bash template for running dataprep and training the model
. ./path.sh || exit 1;

MAIN_PATH=$(pwd)

cd src/

stage=3
stop_stage=3
if [ $# -eq 2 ]; then
    stage=$1
    stop_stage=$2
fi

dumpdir=dump
verbose=0
baseline=cnn
epochs=3
p_m=0.8
q_m=0.1
p_c=0.2
q_c=0.3
stages=2
nodes="4_5"
dataset="mnist"
population_size=10
generations=10


exp_id=exp_stages${stages}_nodes${nodes}
tag="pop_${population_size}_gen_${generations}_epochs_${epochs}_pm${p_m}_qm${q_m}_pc${p_c}_qc${q_c}"

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
    python3 evolution.py  --generations ${generations} \
                        --population_size ${population_size} \
                        --epochs ${epochs} \
                        --exp_path ${EXP_PATH}/${tag}.json \
                        --pm ${p_m} \
                        --qm ${q_m} \
                        --pc ${p_c} \
                        --qc ${q_c} \
                        --cnn_stages ${stages} \
                        --cnn_nodes ${nodes} \
                        --gpu 1 \
                        --dataset ${dataset} \
                        --verbose True \
                #        > ${EXP_PATH}/${tag}.log 
fi

if [ ${stage} -le 3 ] &&[ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Generate plots"
    for set in mnist cifar10 fashion; do
        mkdir -pv ${MAIN_PATH}/plots/${set}
        python3 eval_exp.py --exp_root ${MAIN_PATH}/exp_old/${set} \
                            --dataset ${set}
    done
fi
