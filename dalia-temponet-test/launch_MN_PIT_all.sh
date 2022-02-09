#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

folder="test_30_07"
echo $folder

mkdir $folder

echo

networks="1 2 3 4 5 6 7 8 9 10 11 12 13 "

if [ $1 == 'MN1-PITsmall' ] && [ $2 == '8' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation True --finetuning True --sheet MN1-PITsmall --net_number $net --quantization 8 > ./$folder/cr_val_quant_8_MN1_PITsmall_$net.txt
done
fi

if [ $1 == 'MN1-PITsmall' ] && [ $2 == '4' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation True --finetuning True --sheet MN1-PITsmall --net_number $net --quantization 4 > ./$folder/cr_val_quant_4_MN1_PITsmall_$net.txt
done
fi

if [ $1 == 'MN1-PITsmall' ] && [ $2 == '2' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True --finetuning True --sheet MN1-PITsmall --net_number $net --quantization 2 > ./$folder/cr_val_quant_2_MN1_PITsmall_$net.txt
done
fi


networks="1 2 3 4 5 6 7 8 9 10 11 12 13 "

if [ $1 == 'MN1-PITmed' ] && [ $2 == '8' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation True --finetuning True --sheet MN1-PITmed --net_number $net --quantization 8 > ./$folder/cr_val_quant_8_MN1_PITmed_$net.txt
done
fi

if [ $1 == 'MN1-PITmed' ] && [ $2 == '4' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation True --finetuning True --sheet MN1-PITmed --net_number $net --quantization 4 > ./$folder/cr_val_quant_4_MN1_PITmed_$net.txt
done
fi

if [ $1 == 'MN1-PITmed' ] && [ $2 == '2' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True --finetuning True --sheet MN1-PITmed --net_number $net --quantization 2 > ./$folder/cr_val_quant_2_MN1_PITmed_$net.txt
done
fi