#!/bin/sh

for lambda in 10
do 
    nice -n 7 python src/main.py --mode dim --lmbda $lambda --order 3 -v 1 -max_cycles 40 -cycle_length 10 -warmup_cycles 2 -warmup_cycle_length 10 -n_betas 3 -swap_frequency 3 -n_reps 3
done
