#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

echo "Test TimePPG/TEMPONet search mix: 05/07/2021"

networks="mixTempoNet_layer_248_layers  mixTimePPG_big_248_layers mixTimePPG_medium_248_layers  mixTimePPG_small_248_layers "
for net in $networks;
do
	echo $net
	echo "cd 000035"
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation False -a $net  --cd 0.000035 > ./test_05_07/cr_val_$net\_000035.txt
done

for net in $networks;
do
	echo $net
	echo "cd 00035"
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation False -a $net  --cd 0.00035 > ./test_05_07/cr_val_$net\_00035.txt
done


for net in $networks;
do
	echo $net
	echo "cd 0035"
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation False -a $net  --cd 0.0035 > ./test_05_07/cr_val_$net\_0035.txt
done
