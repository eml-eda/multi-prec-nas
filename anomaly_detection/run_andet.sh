#!/usr/bin/env bash

strength=$1
warmup=$2
path="."
arch=$3
project="multi-precision-nas_andet"

tags="init_same warmup8bit reg_w"

mkdir -p ${arch}
mkdir -p ${arch}/model_${strength}

if [[ "$4" == "search" ]]; then
    echo Search
    split=0.2
    python3 search.py ${arch}/model_${strength} -a mix${arch} \
        -d toy_car --arch-data-split ${split} \
        --epochs 100 --step-epoch 20 -b 512 --warmup ${warmup} --warmup-8bit \
        --lr 0.001 --lra 0.01 --wd 1e-4 \
        --ai same --cd ${strength} --rt weights \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 5 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} --debug | tee ${arch}/model_${strength}/log_search_${strength}.txt
fi

if [[ "$5" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${arch}/model_${strength} -a quant${arch} -d toy_car \
        --epochs 100 --step-epoch 20 -b 512 --patience 100 \
        --lr 0.001 --wd 0 --cd ${strength} \
        --seed 42 --gpu 0 \
        --ac ${arch}/model_${strength}/arch_model_best.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${arch}/model_${strength}/log_finetune_$strength.txt
else
    echo From-Scratch
    python3 main.py ${arch}/model_${strength} -a quant${arch} -d toy_car \
        --epochs 100 --step-epoch 20 -b 512 --patience 100 \
        --lr 0.001 --wd 0 \
        --seed 42 --gpu 0 \
        --ac ${arch}/model_${strength}/arch_model_best.pth.tar \
        --visualization -pr ${project} --tags ${tags} | tee ${arch}/model_${strength}/log_fromscratch_$strength.txt
fi
