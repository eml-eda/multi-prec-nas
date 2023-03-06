**NEW RELEASE: we released our new, engineered and user-friendly DNAS library named [PLiNIO](https://github.com/eml-eda/plinio) which includes channel-wise precision assignement among the different implemented methods. We highly suggest to try this new release for your experiments!**


Copyright (C) 2022 Politecnico di Torino, Italy. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Matteo Risso, Alessio Burrello, Luca Benini, Enrico Macii, Massimo Poncino, Daniele Jahier Pagliari

# multi-prec-nas

## Reference
If you use our code in your experiments, please make sure to cite our paper:
```
@INPROCEEDINGS{9969373,
  author={Risso, Matteo and Burrello, Alessio and Benini, Luca and Macii, Enrico and Poncino, Massimo and Pagliari, Daniele Jahier},
  booktitle={2022 IEEE 13th International Green and Sustainable Computing Conference (IGSC)}, 
  title={Channel-wise Mixed-precision Assignment for DNN Inference on Constrained Edge Nodes}, 
  year={2022},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/IGSC55832.2022.9969373}}
```
## Datasets
The current version support the following datasets and tasks taken from the benchmark suite MLPerf Tiny:
- CIFAR10 - Image Classification.
- MSCOCO - Visual Wake Words.
- Google Speech Commands v2 - Keyword Spotting.
- ToyADMOS - Anomaly Detection

## How to run
### Image Classification
1. Visit the folder: `cd image_classification`.
2. Run the provided shell script `run_ic.sh`: 
```
source run_ic.sh <regularization_strenght> 0 resnet8_w248a248_multiprec search ft
```

### Visual Wake Words
1. Run the provided `Makefile` to download the desired dataset: `make vww-init`.
2. Visit the folder: `cd visual_wake_words`.
3. Run the provided shell script `run_vww.sh`: 
```
source run_vww.sh <regularization_strenght> 0 mobilenetv1_w248a248_multiprec search ft
```

### Keyword Spotting
1. Visit the folder: `cd keyword_spotting`.
2. Run the provided shell script `run_kws.sh`: 
```
source run_kws.sh <regularization_strenght> 0 dscnn_w248a248_multiprec search ft
```

### Anomaly Detection
1. Run the provided `Makefile` to download the desired dataset: `make andet-init`.
2. Visit the folder: `cd anomaly_detection`.
3. Run the provided shell script `run_andet.sh`: 
```
source run_andet.sh <regularization_strenght> 0 denseae_w248a248_multiprec search ft
```

## License
This code is released under Apache 2.0, see the LICENSE file in the root of this repository for details.
