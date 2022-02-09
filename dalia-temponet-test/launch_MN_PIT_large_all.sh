#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

folder="test_02_08"
echo $folder

mkdir $folder

echo

networks="7 8 9 10 11 12 13 14 15 "

if [ $1 == 'MN1-PITlarge' ] && [ $2 == '8' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation True --finetuning True --sheet MN1-PITlarge --net_number $net --quantization 8 > ./$folder/cr_val_quant_8_MN1_PITlarge_$net.txt
done
fi

if [ $1 == 'MN1-PITlarge' ] && [ $2 == '4' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation True --finetuning True --sheet MN1-PITlarge --net_number $net --quantization 4 > ./$folder/cr_val_quant_4_MN1_PITlarge_$net.txt
done
fi

if [ $1 == 'MN1-PITlarge' ] && [ $2 == '2' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True --finetuning True --sheet MN1-PITlarge --net_number $net --quantization 2 > ./$folder/cr_val_quant_2_MN1_PITlarge_$net.txt
done
fi


networks="7 8 9 10 11 12 13 14 15 16 17 18 19 "

if [ $1 == 'MN1-PITlargest' ] && [ $2 == '8' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation True --finetuning True --sheet MN1-PITlargest --net_number $net --quantization 8 > ./$folder/cr_val_quant_8_MN1_PITlargest_$net.txt
done
fi

if [ $1 == 'MN1-PITlargest' ] && [ $2 == '4' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation True --finetuning True --sheet MN1-PITlargest --net_number $net --quantization 4 > ./$folder/cr_val_quant_4_MN1_PITlargest_$net.txt
done
fi

if [ $1 == 'MN1-PITlargest' ] && [ $2 == '2' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True --finetuning True --sheet MN1-PITlargest --net_number $net --quantization 2 > ./$folder/cr_val_quant_2_MN1_PITlargest_$net.txt
done
fi