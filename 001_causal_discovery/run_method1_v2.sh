#!/bin/bash

#for dataset in xyguassian yxguassian balancescale krkp waveform splice cifar10 cifar10n_worst cifar10n_aggre cifar10n_random1 cifar10n_random2 cifar10n_random3
for dataset in cifar10n_random1 cifar10n_random2 cifar10n_random3
do
    #for nt in sym instance pair
    for nt in instance pair sym
    do
        for seed in 1 2 3 4 5
        do
            for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
            do
                python3 main_method1_v2.py --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --flip_rate_fixed ${rate}
            done             
        done
    done
done
