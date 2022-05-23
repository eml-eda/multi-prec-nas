#!/usr/bin/env bash

source run_kws.sh 1e-84 0 dscnn_w8a4 nsearch nft
source run_kws.sh 1e-44 0 dscnn_w4a4 nsearch nft
source run_kws.sh 1e-24 0 dscnn_w2a4 nsearch nft

source run_kws.sh 1e-82 0 dscnn_w8a2 nsearch nft
source run_kws.sh 1e-42 0 dscnn_w4a2 nsearch nft
source run_kws.sh 1e-22 0 dscnn_w2a2 nsearch nft