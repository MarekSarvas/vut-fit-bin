#!/usr/bin/bash
# Bash template for running dataprep and training the model
#$ -N run 
#$ -q long.q@*
#$ -l gpu=1,gpu_ram=8G,ram_free=16G,mem_free=16G

#$ -o /mnt/matylda4/xsarva00/fit_bin/vut-fit-bin/exp/exp1_not_training.log
#$ -e /mnt/matylda4/xsarva00/fit_bin/vut-fit-bin/exp/exp1_not_training.err

cd /mnt/matylda4/xsarva00/fit_bin/vut-fit-bin
. ./path.sh || exit 1;

MAIN_PATH=$(pwd)

cd src/

stage=2
stop_stage=2
dumpdir=dump
verbose=0

baseline=cnn
epochs=10
exp_id='exp1_test'
tag='exp1_test'

EXP_PATH=${MAIN_PATH}/exp/${exp_id}

mkdir -pv ${EXP_PATH}

if [ ${stage} -le 0 ] &&[ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"
    rm -r ../data/MNIST/
    python download_data.py
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
    python3 evolution.py  --generations 20 \
                        --population_size 10 \
                        --epochs 4 \
                        --exp_path ${EXP_PATH}/${tag} \
                        --mut_p 0.8 \
                        --cross_p 0.2 \
                        --cnn_stages 2 \
                        --cnn_nodes "4,5" \
                        --gpu 1 \
                        --verbose True \
                #        > ${EXP_PATH}/${tag}.log 
fi

