#!/usr/bin/env bash
strength=$1
warmup=$2
path="/space/risso/multi_prec_exp"
# arch: {mix, quant}mobilenetv1_w248a248_multiprec, {mix, quant}mobilenetv1_w0248a248_multiprec
arch="denseae_fp"
#arch="dscnn_w4a8"
project="multi-Precision-nas_andet"

#tags="warmup"
tags="fp"
#tags="w8a8"
#tags="init_same no_wp reg_w"

mkdir -p ${path}/an_det
mkdir -p ${path}/an_det/${arch}
mkdir -p ${path}/an_det/${arch}/model_${strength}

if [[ "$3" == "search" ]]; then
    echo Search
    python3 anomaly_detection/search.py ${path}/an_det/${arch}/model_${strength} -a mix${arch} -d toy_car --epochs 100 --step-epoch 10 -b 512 --warmup ${warmup}\
        --lr 0.001 --lra 0.01 --wd 1e-4 \
        --ai same --cd $strength --rt weights \
        --seed 42 --gpu 0 \
        --visualization -pr ${project} --tags ${tags} --debug | tee ${path}/an_det/${arch}/model_${strength}/log_search_$strength.txt
fi

if [[ "$4" == "ft" ]]; then
    echo Fine-Tune
    python3 anomaly_detection/main.py ${path}/an_det/${arch}/model_${strength} -a quant${arch} -d toy_car --epochs 100 --step-epoch 10 -b 512 \
        --lr 0.001 --wd 5e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/an_det/${arch}/model_${strength}/arch_checkpoint.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/an_det/${arch}/model_${strength}/log_finetune_$strength.txt
else
    echo From-Scratch
    python3 anomaly_detection/main.py ${path}/an_det/${arch}/model_${strength} -a quant${arch} -d toy_car \
        --epochs 200 --step-epoch 10 -b 512 \
        --lr 0.001 --wd 0 \
        --gpu 0 \
        --ac ${path}/an_det/${arch}/model_${strength}/arch_checkpoint.pth.tar | tee ${path}/an_det/${arch}/model_${strength}/log_fromscratch_$strength.txt
fi