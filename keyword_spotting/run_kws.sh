#!/usr/bin/env bash
strength=$1
warmup=$2
path="/space/risso/multi_prec_exp/kws"
# arch: {mix, quant}mobilenetv1_w248a248_multiprec, {mix, quant}mobilenetv1_w0248a248_multiprec
#arch="dscnn_fp"
#arch="dscnn_w8a8"
arch="dscnn_w248a8_multiprec"
project="multi-Precision-nas_kws"

#tags="warmup"
#tags="fp"
#tags="w8a8"
tags="init_same no_wp reg_w softemp"

mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/model_${strength}

export WANDB_MODE=offline

if [[ "$3" == "search" ]]; then
    echo Search
    split=0.2
    python3 search.py ${path}/${arch}/model_${strength} -a mix${arch} \
        -d GoogleSpeechCommands --arch-data-split ${split} \
        --epochs 300 --step-epoch 10 -b 100 --warmup ${warmup} --warmup-8bit \
        --lr 0.00001 --lra 0.01 --wd 1e-4 \
        --ai same --cd ${strength} --rt weights \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 5 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} --debug | tee ${path}/${arch}/model_${strength}/log_search_$strength.txt
fi

if [[ "$4" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${path}/${arch}/model_${strength} -a quant${arch} \
        -d GoogleSpeechCommands --epochs 300 --step-epoch 10 -b 100 \
        --lr 0.00001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/${arch}/model_${strength}/arch_model_best.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/log_finetune_$strength.txt
elif [[ "$4" == "eval" ]]; then
    echo Eval
    python3 main.py ${path}/${arch}/model_${strength} -a quant${arch} \
        -d GoogleSpeechCommands --epochs 300 --step-epoch 10 -b 100 \
        --lr 0.00001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/${arch}/model_${strength}/arch_model_best.pth.tar -ft \
        --evaluate
else
    echo From-Scratch
    python3 main.py ${path}/${arch}/model_${strength} -a quant${arch} -d GoogleSpeechCommands \
        --epochs 300 --step-epoch 10 -b 100 \
        --lr 0.00001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/${arch}/model_${strength}/arch_checkpoint.pth.tar | tee ${path}/${arch}/model_${strength}/log_fromscratch_$strength.txt
fi