#!/usr/bin/env bash

strenght=( 5.0e+1 7.5e+0 2.5e+0 )

for s in "${strenght[@]}"
do
    echo "Strength: ${s}"
    source run_kws_local.sh ${s} 0 search ft
done