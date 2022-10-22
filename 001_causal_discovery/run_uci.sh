#!/bin/bash


for dataset in splice balancescale krkp splice waveform
do
    for nt in sym instance pair
    do
        for seed in 1
        do

            for rate in 0.2 0.4 
            do
                python3 main_uci.py --flip_rate_fixed ${rate} --seed ${seed}  --noise_type ${nt} --dataset ${dataset} > ${dataset}_${nt}_flip_rate${rate}_seed${seed}.out
            done

        done
    done
done

