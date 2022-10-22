#!/bin/bash


for dataset in xyguassian yxguassian balancescale krkp waveform splice
do
    for nt in sym instance pair
    do
        for seed in 1
        do

            for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
            do
                for pca in 5 3 1
                do
                    for near_perc in 0.1
                    do
                        for sample_size in 100
                        do
                            python3 main_PCA.py --sample_size ${sample_size} --near_percentage ${near_perc} --pca_k ${pca} --flip_rate_fixed ${rate} --seed ${seed}  --noise_type ${nt} --dataset ${dataset} 
                        done
                    done
                done
            done

        done
    done
done

