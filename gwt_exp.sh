#!/usr/bin/env bash

strength=$1
path="/space/risso/multi_prec_exp"
# arch: {mix, quant}res18_w248a248_multiprec, {mix, quant}res18_w0248a248_multiprec
arch=res18_w2345678a8_chan
#arch=res18_w8a8
project=gwt

mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/model_${strength}

if [[ "$2" == "search" ]]; then
    echo Search
    python3 search.py ${path}/${arch}/model_${strength} -a mix${arch} -d cifar --epochs 100 --step-epoch 10 -b 128 \
        --lr 0.1 --lra 0.01 --wd 1e-4 \
        --cd $strength --seed 42 --gpu 0 \
        --visualization -pr ${project} --tags no-bias32b | tee ${path}/${arch}/model_${strength}/log_search_$strength.txt
fi

if [[ "$3" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${path}/${arch}/model_${strength} -a quant${arch} -d cifar --epochs 100 --step-epoch 10 -b 128 \
        --lr 0.1 --wd 5e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/${arch}/model_${strength}/arch_checkpoint.pth.tar -ft \
        --visualization -pr ${project} --tags no-bias32b | tee ${path}/${arch}/model_${strength}/log_finetune_$strength.txt
else
    echo From-Scratch
    python3 main.py ${path}/${arch}/model_${strength} -a quant${arch} -d cifar --epochs 100 --step-epoch 10 -b 128 \
        --lr 0.1 --wd 5e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/${arch}/model_${strength}/arch_checkpoint.pth.tar | tee ${path}/${arch}/model_${strength}/log_fromscratch_$strength.txt
fi