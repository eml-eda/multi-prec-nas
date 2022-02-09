#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

folder="test_mixed"
echo $folder

mkdir $folder
if [ $3 == '0.0001' ]
then
python3 test_TEMPONetDaliaTrainer.py --gpu $4 --cross-validation False --finetuning False --sheet $1 --cd $3 --net_number $2 --quantization mix-search > ./$folder/cr_val_$1\_$2\_cd_med.txt
fi

if [ $3 == '0.00001' ]
then
python3 test_TEMPONetDaliaTrainer.py --gpu $4 --cross-validation False --finetuning False --sheet $1 --cd $3 --net_number $2 --quantization mix-search > ./$folder/cr_val_$1\_$2\_cd_small.txt
fi

if [ $3 == '0.001' ]
then
python3 test_TEMPONetDaliaTrainer.py --gpu $4 --cross-validation False --finetuning False --sheet $1 --cd $3 --net_number $2 --quantization mix-search > ./$folder/cr_val_$1\_$2\_cd_big.txt
fi