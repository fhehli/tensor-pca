#!/bin/sh

for dim in 125
do 
    nice -n 7 python main.py --mode lambda --dim $dim --order 4 -v 1
done
