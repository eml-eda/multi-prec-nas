#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

folder="test_mixed_finetune"
echo $folder

mkdir $folder

python3 test_TEMPONetDaliaTrainer.py --gpu $4 --cross-validation True --finetuning True --sheet $1 --net_number $2 --quantization mix --cd $3 > ./$folder/cr_val_$1\_$2\_cd_$3.txt
