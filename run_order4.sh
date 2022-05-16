screen -S pt

for dim in 25 50 100 125
do 
    if [ $dim == 125 ]; then
        verbose=1
    else
        verbose=0
    fi
    nice -n 7 python main.py --dim $dim -v $verbose &
done