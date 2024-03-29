#!/usr/bin/env bash

strength=$1
warmup=$2
path="."
arch=$3

project="multi-precision-nas_vww"

tags="init_same warmup reg_w"

mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/model_${strength}

if [[ "$4" == "search" ]]; then
    echo Search
    split=0.2
    python3 search.py ${path}/${arch}/model_${strength} -a mix${arch} \
        -d coco2014_96_tf --arch-data-split ${split} --patience 70 \
        --epochs 70 --step-epoch 10 -b 32 --warmup ${warmup} --warmup-8bit \
        --lr 0.001 --lra 0.01 --wd 1e-4 \
        --ai same --cd ${strength} --rt weights \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 5 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} --debug | tee ${path}/${arch}/model_${strength}/log_search_${strength}.txt
fi

if [[ "$5" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${arch}/model_${strength} -a quant${arch} -d coco2014_96_tf \
        --epochs 70 --step-epoch 10 -b 32 \
        --lr 0.001 --wd 1e-4 --cd ${strength} \
        --seed 42 --gpu 0 \
        --ac ${path}/${arch}/model_${strength}/arch_checkpoint.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/log_finetune_${strength}.txt
elif [[ "$4" == "eval" ]]; then
    echo Eval
    python3 main.py ${path}/vww/${arch}/model_${strength} -a quant${arch} -d coco2014_96_tf \
        --epochs 70 --step-epoch 10 -b 32 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/vww/${arch}/model_${strength}/model_best.pth.tar -ft \
        --evaluate
else
    echo From-Scratch
    python3 main.py ${path}/${arch}/model_${strength} -a quant${arch} -d coco2014_96_tf --epochs 70 --step-epoch 10 \
        -b 32 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/${arch}/model_${strength}/arch_model_best.pth.tar | tee ${path}/${arch}/model_${strength}/log_fromscratch_${strength}.txt
fi