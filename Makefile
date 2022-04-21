# Makefile

SHELL := /bin/bash

URL1="https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1"
ZIPFILE="dev_data_ToyCar.zip"
URL2="https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip?download=1"

default:
	echo "run make followed by benchmark name + -init"
	echo "E.g., make image_classification-init"

andet-init:	
	mkdir -p anomaly_detection/dev_data
	curl ${URL1} -o ${ZIPFILE} || wget $(URL1) -O $(ZIPFILE)
	unzip ${ZIPFILE} -d anomaly_detection/dev_data
	rm ${ZIPFILE}
	curl ${URL2} -o ${ZIPFILE} || wget ${URL2} -O ${ZIPFILE}
	unzip ${ZIPFILE} -d dev_data
	rm ${ZIPFILE}

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

clean:
	rm -rf data
	rm -rf __pycache__
	rm -rf cifar10/__pycache__
	rm -rf kws/__pycache__
	rm -rf mnist/__pycache__
	rm -rf models/__pycache__
	rm -rf vww/__pycache__