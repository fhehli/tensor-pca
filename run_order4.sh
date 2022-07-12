#!/bin/sh

for dim in 50 75 100 125 150
do 
    nice -n 7 python main.py --dim $dim -v 1
done
