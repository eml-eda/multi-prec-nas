#!/usr/bin/env bash

strength=$1
warmup=$2
path="/space/risso/multi_prec_exp"

## arch: {mix, quant}res18_w248a248_multiprec, {mix, quant}res18_w0248a248_multiprec
#arch="res18_w0248a248_multiprec"
arch="res18_w0248a8_multiprec"
#arch="res18_w248a8_multiprec"
#arch="res18_w0248a4_multiprec"
#arch="res18_w248a8_chan"
#arch="res18_w248a8_chan_mp"
#arch="res8_fp"
#arch="res18_w2a8"
#arch="res8_w248a8_chan"
#arch="res18_fp"

project="multi-precision-nas_ic"

#tags="warmup"
#tags="fp"
#tags="init_same no_wp reg_w"
tags="init_same warmup reg_w"

mkdir -p saved_models/${arch}
mkdir -p saved_models/${arch}/model_${strength}

if [[ "$3" == "search" ]]; then
    echo Search
    python3 search.py data -a mix${arch} -d cifar --arch-data-split 0.2 \
        --epochs 100 --step-epoch 10 -b 128 --warmup ${warmup} \
        --lr 0.1 --lra 0.01 --wd 1e-4 \
        --ai same --cd ${strength} --rt weights \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 5 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} --debug | tee log/${arch}/model_${strength}/log_search_${strength}.txt
fi

if [[ "$4" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py data -a quant${arch} -d cifar --epochs 100 --step-epoch 10 -b 128 \
        --lr 0.1 --wd 5e-4 \
        --seed 42 --gpu 0 \
        --ac saved_models/${arch}/model_${strength}/arch_model_best.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee log/${arch}/model_${strength}/log_finetune_${strength}.txt
else
    echo From-Scratch
    python3 main.py data -a quant${arch} -d cifar --epochs 100 --step-epoch 10 -b 128 \
        --lr 0.1 --wd 5e-4 \
        --seed 42 --gpu 0 \
        --ac saved_models/${arch}/model_${strength}/arch_checkpoint.pth.tar | tee log/${arch}/model_${strength}/log_fromscratch_$strength.txt
fi