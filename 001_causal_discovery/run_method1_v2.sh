#!/bin/bash

#for dataset in xyguassian yxguassian balancescale krkp waveform splice cifar10 cifar10n_worst cifar10n_aggre cifar10n_random1 cifar10n_random2 cifar10n_random3
for dataset in g241c
#for dataset in secstr
do
    for nt in sym instance pair
    do
        for seed in 1 2 3 4 5
        do
            for rate in 0.01 0.2 0.4
            do
                python3 main_method1_v2.py --output "g241c" --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --flip_rate_fixed ${rate}
                 
            done             
        done
    done
done
h