#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

echo "Test TimePPG/TEMPONet quantized: 05/07/2021"

mkdir test_05_07
networks="TempoNet_multiprec_big TempoNet_multiprec_medium TempoNet_multiprec_small TimePPG_big_multiprec_big TimePPG_big_multiprec_medium TimePPG_big_multiprec_small TimePPG_medium_multiprec_big TimePPG_medium_multiprec_medium TimePPG_medium_multiprec_small TimePPG_small_multiprec_big TimePPG_small_multiprec_medium TimePPG_small_multiprec_small"

for net in $networks;
do
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True -a $net --finetuning True > ./test_05_07/cr_val_$net.txt
done