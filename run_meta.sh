#!/usr/bin/bash


stage=0
stop_stage=2
dumpdir=dump
verbose=0

baseline=cnn

epochs=5
p_m=0.8
q_m=0.1
p_c=0.2
q_c=0.3
stages=3
nodes="3_4_5"
dataset="cifar10"
population_size=20
generations=30


exp_id=exp_stages${stages}_nodes${nodes}
tag="pop_${population_size}_gen_${generations}_epochs_${epochs}_pm${p_m}_qm${q_m}_pc${p_c}_qc${q_c}"


EXP_PATH=${BASE}/vut-fit-bin/exp_meta/${dataset}/${exp_id}



mkdir -pv ${EXP_PATH}
cd src/

if [ ${stage} -le 0 ] &&[ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"
    python download_data.py --data_path ../data
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

