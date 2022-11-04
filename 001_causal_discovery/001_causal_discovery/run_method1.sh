#!/bin/bash

#for dataset in xyguassian yxguassian balancescale krkp waveform splice
for dataset in balancescale waveform
do
    #for nt in sym instance pair
    for nt in sym
    do
        for seed in 1 
        do
            for rate in 0.1 
            do
                python3 main_method1.py --seed ${seed}  --noise_type ${nt} --dataset ${dataset} --flip_rate_fixed ${rate}
            done             
        done
    done
done
