# Makefile

SHELL := /bin/bash

default:
	echo "run make followed by benchmark name + -init"
	echo "E.g., make image_classification-init"

image_classification-init:
	mkdir -p image_classification/log
	mkdir -p image_classification/saved_models
	mkdir -p image_classification/data

kws-init:
	mkdir -p kws/log
	mkdir -p kws/saved_models
	mkdir -p kws/data

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