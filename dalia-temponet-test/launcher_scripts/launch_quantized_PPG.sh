#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

echo "Test TimePPG/TEMPONet quantized: 05/07/2021"

mkdir test_05_07
networks="TempoNetfloat TempoNet_layer_quantized_16 TempoNet_layer_quantized_2 TempoNet_layer_quantized_4 TempoNet_layer_quantized_8 TempoNet_layer_quantized_big TempoNet_layer_quantized_medium TempoNet_layer_quantized_small TimePPG_big_quantized_2 TimePPG_big_quantized_4 TimePPG_big_quantized_8 TimePPG_big_quantized_big TimePPG_big_quantized_medium TimePPG_big_quantized_small TimePPG_medium_quantized_2 TimePPG_medium_quantized_4 TimePPG_medium_quantized_8 TimePPG_medium_quantized_big TimePPG_medium_quantized_medium TimePPG_medium_quantized_small TimePPG_small_quantized_2 TimePPG_small_quantized_4 TimePPG_small_quantized_8 TimePPG_small_quantized_big TimePPG_small_quantized_medium TimePPG_small_quantized_small TimePPGfloat_big TimePPGfloat_medium TimePPGfloat_small"

for net in $networks;
do
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 0 --cross-validation True -a $net --finetuning True > ./test_05_07/cr_val_$net.txt
done