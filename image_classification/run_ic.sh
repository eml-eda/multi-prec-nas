#!/usr/bin/env bash

strength=$1
warmup=$2
path="."

#arch="res8_fp"
arch="res8_w8a8"
#arch="res8_w248a8_chan"

project="multi-precision-nas_ic"

#tags="warmup"
#tags="fp"
#tags="init_same no_wp reg_w"
tags="init_same wp reg_w softemp"

mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/model_${strength}

export WANDB_MODE=offline

if [[ "$3" == "search" ]]; then
    echo Search
    split=0.2
    python3 search.py data -a mix${arch} \
        -d cifar --arch-data-split ${split} \
        --epochs 500 --step-epoch 50 -b 32 \
        --warmup ${warmup} --warmup-8bit --patience 100 \
        --lr 0.001 --lra 0.01 --wd 1e-4 \
        --ai same --cd ${strength} --rt weights \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 5 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} --debug | tee ${path}/${arch}/model_${strength}/log_search_${strength}.txt
fi

if [[ "$4" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py data -a quant${arch} \
        -d cifar --epochs 500 --step-epoch 50 -b 32 --patience 100 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${arch}/model_${strength}/arch_model_best.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/log_finetune_${strength}.txt
else
    echo From-Scratch
    python3 main.py data -a quant${arch} \
        -d cifar --epochs 500 --step-epoch 10 -b 32 --patience 100 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${arch}/model_${strength}/arch_model_best.pth.tar | tee ${path}/${arch}/model_${strength}/log_fromscratch_${strength}.txt
fi