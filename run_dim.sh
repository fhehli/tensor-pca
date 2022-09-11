#!/bin/sh

for lambda in 2 5 10
do 
    nice -n 7 python main.py --mode dim --lmbda $lambda --order 4 -max_cycles 200 -v 1 -n_reps 10  
done
