#!/usr/bin/env bash
strength=$1
warmup=$2
path="/space/risso/multi_prec_exp"
# arch: {mix, quant}mobilenetv1_w248a248_multiprec, {mix, quant}mobilenetv1_w0248a248_multiprec
#arch="dscnn_fp"
arch="dscnn_w2a8"
project="multi-Precision-nas_kws"

#tags="warmup"
#tags="fp"
tags="w8a8"
#tags="init_same no_wp reg_w"

mkdir -p ${path}/kws
mkdir -p ${path}/kws/${arch}
mkdir -p ${path}/kws/${arch}/model_${strength}

if [[ "$3" == "search" ]]; then
    echo Search
    python3 keyword_spotting/search.py ${path}/kws/${arch}/model_${strength} -a mix${arch} -d coco2014_96 --epochs 100 --step-epoch 10 -b 16 --warmup ${warmup}\
        --lr 0.1 --lra 0.01 --wd 1e-4 \
        --ai same --cd $strength --rt weights \
        --seed 42 --gpu 0 \
        --visualization -pr ${project} --tags ${tags} --debug | tee ${path}/kws/${arch}/model_${strength}/log_search_$strength.txt
fi

if [[ "$4" == "ft" ]]; then
    echo Fine-Tune
    python3 keyword_spotting/main.py ${path}/kws/${arch}/model_${strength} -a quant${arch} -d coco2014_96 --epochs 100 --step-epoch 10 -b 16 \
        --lr 0.1 --wd 5e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/kws/${arch}/model_${strength}/arch_checkpoint.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/kws/${arch}/model_${strength}/log_finetune_$strength.txt
else
    echo From-Scratch
    python3 keyword_spotting/main.py ${path}/kws/${arch}/model_${strength} -a quant${arch} -d GoogleSpeechCommands \
        --epochs 300 --step-epoch 10 -b 100 \
        --lr 0.00001 --wd 1e-4 \
        --gpu 0 \
        --ac ${path}/kws/${arch}/model_${strength}/arch_checkpoint.pth.tar | tee ${path}/kws/${arch}/model_${strength}/log_fromscratch_$strength.txt
fi