#!/bin/sh

for lambda in 10 20 50
do 
    nice -n 7 python main.py --mode dim --lmbda $lambda --order 3 -v 1 -max_cycles 40 -cycle_length 10 -warmup_cycles 2 -warmup_cycle_length 10 -n_betas 3 -swap_frequency 3 -n_reps 5 
done
