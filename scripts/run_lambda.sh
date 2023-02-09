#!/bin/sh

for dim in 100 150 200 250 300 250 400 450 500
do 
    nice -n 7 python src/main.py --mode lambda --dim $dim --order 3 -v 1
done
