#!/usr/bin/bash
# Bash template for running dataprep and training the model
. ./path.sh || exit 1;
cd src/

stage=1
stop_stage=1
dumpdir=dump
verbose=0

baseline=cnn
epochs=10
tag=

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
    echo "stage 2: Neuroevolution I guess"
fi

