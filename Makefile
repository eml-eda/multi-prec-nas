# Makefile

SHELL := /bin/bash

URL1_andet="https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1"
ZIPFILE_andet="dev_data_ToyCar.zip"
URL2_andet="https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip?download=1"

default:
	echo "run make followed by benchmark name + -init"
	echo "E.g., make image_classification-init"

andet-init:	
	mkdir -p anomaly_detection/dev_data
	curl ${URL1_andet} -o ${ZIPFILE_andet} || wget $(URL1_andet) -O $(ZIPFILE_andet)
	unzip ${ZIPFILE_andet} -d anomaly_detection/dev_data
	rm ${ZIPFILE_andet}
	curl ${URL2_andet} -o ${ZIPFILE} || wget ${URL2_andet} -O ${ZIPFILE_andet}
	unzip ${ZIPFILE_andet} -d anomaly_detection/dev_data
	rm ${ZIPFILE_andet}

image_classification-init:
	mkdir -p image_classification/log
	mkdir -p image_classification/saved_models
	mkdir -p image_classification/data

kws-init:
	echo "nothing to do"

vww-init:
	wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz -P visual_wake_words
	tar -xvf visual_wake_words/vw_coco2014_96.tar.gz -C visual_wake_words
	rm visual_wake_words/vw_coco2014_96.tar.gz
