#!/usr/bin/env bash

strength=$1
warmup=$2
path="/space/risso/multi_prec_exp"
# arch: {mix, quant}mobilenetv1_w248a248_multiprec, {mix, quant}mobilenetv1_w0248a248_multiprec
#arch="mobilenetv1_fp"
#arch="mobilenetv1_w2a8"
#arch="mobilenetv1_w4a8"
#arch="mobilenetv1_w8a8"
arch="mobilenetv1_w0248a8_multiprec"
#arch="mobilenetv1_w248a8_multiprec"
#arch="mobilenetv1_w248a8_chan"
#arch="mobilenetv1_w248a8_chan_mp"
project="multi-Precision-nas_vww"

#tags="warmup"
#tags="fp"
tags="init_same warmup reg_w gumbel"
#tags="init_same warmup reg_w"
#tags="init_same warmup reg_w ft_as_mp"

mkdir -p ${path}/vww
mkdir -p ${path}/vww/${arch}
mkdir -p ${path}/vww/${arch}/model_${strength}

if [[ "$3" == "search" ]]; then
    echo Search
    python3 visual_wake_words/search.py ${path}/vww/${arch}/model_${strength} -a mix${arch} -d coco2014_96_tf \
        --epochs 70 --step-epoch 10 -b 32 --warmup ${warmup}\
        --lr 0.001 --lra 0.01 --wd 1e-4 \
        --ai same --cd $strength --rt weights \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 0.1 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} --debug | tee ${path}/vww/${arch}/model_${strength}/log_search_$strength.txt
        #--visualization -pr ${project} --tags ${tags} --debug | tee ${path}/vww/${arch}/model_${strength}/log_search_$strength.txt
fi

if [[ "$4" == "ft" ]]; then
    echo Fine-Tune
    python3 visual_wake_words/main.py ${path}/vww/${arch}/model_${strength} -a quant${arch} -d coco2014_96_tf \
        --epochs 70 --step-epoch 10 -b 32 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/vww/${arch}/model_${strength}/arch_checkpoint.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/vww/${arch}/model_${strength}/log_finetune_$strength.txt
elif [[ "$4" == "eval" ]]; then
    echo Eval
    python3 visual_wake_words/main.py ${path}/vww/${arch}/model_${strength} -a quant${arch} -d coco2014_96_tf \
        --epochs 70 --step-epoch 10 -b 32 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/vww/${arch}/model_${strength}/model_best.pth.tar -ft \
        --evaluate
else
    echo From-Scratch
    python3 visual_wake_words/main.py ${path}/vww/${arch}/model_${strength} -a quant${arch} -d coco2014_96_tf --epochs 70 --step-epoch 10 \
        -b 32 \
        --lr 0.001 --wd 1e-4 \
        --gpu 0 \
        --ac ${path}/vww/${arch}/model_${strength}/arch_model_best.pth.tar | tee ${path}/vww/${arch}/model_${strength}/log_fromscratch_$strength.txt
fi