#!/usr/bin/env bash
# Filename: launch_regression_test
# gap_sdk
# Total time: 7.19 min on laptop

folder="test_28_07"
echo $folder

mkdir $folder

echo

networks="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 "

if [ $1 == 'MN-dilht' ] && [ $2 == '8' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation True --finetuning True --sheet MN-dilht --net_number $net --quantization 8 > ./$folder/cr_val_quant_8_MN_dilht_$net.txt
done
fi

if [ $1 == 'MN-dilht' ] && [ $2 == '4' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation True --finetuning True --sheet MN-dilht --net_number $net --quantization 4 > ./$folder/cr_val_quant_4_MN_dilht_$net.txt
done
fi

if [ $1 == 'MN-dilht' ] && [ $2 == '2' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True --finetuning True --sheet MN-dilht --net_number $net --quantization 2 > ./$folder/cr_val_quant_2_MN_dilht_$net.txt
done
fi


networks="1 2 3 4 5 6 7 8 9 10 11 "

if [ $1 == 'PIT' ] && [ $2 == '8' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 1 --cross-validation True --finetuning True --sheet PIT --net_number $net --quantization 8 > ./$folder/cr_val_quant_8_PIT_$net.txt
done
fi

if [ $1 == 'PIT' ] && [ $2 == '4' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation True --finetuning True --sheet PIT --net_number $net --quantization 4 > ./$folder/cr_val_quant_4_PIT_$net.txt
done
fi

if [ $1 == 'PIT' ] && [ $2 == '2' ]
then
for net in $networks;
do
	echo $1 $2
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True --finetuning True --sheet PIT --net_number $net --quantization 2 > ./$folder/cr_val_quant_2_PIT_$net.txt
done
fi