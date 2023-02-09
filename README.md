# Parallel Tempering for Tensor PCA
This repo contains the code I wrote for my bachelor thesis. You can have a look at the thesis [here](https://drive.google.com/file/d/1t4w060kvyQq8v48RNOUN5ScK0dxDJrQ_/view?usp=sharing).


## How to run
If you want to rerun the experiments, you can run any of the shell scripts in `/scripts`. Alternatively, you can contact me for the data I collected.

### Environment
With [`miniconda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages) installed, run
```
conda env create -f env.yml
conda activate parallel-tempering
```

### `screen`
When running the code on the D-MATH central clients, use [`screen`](https://blogs.ethz.ch/isgdmath/screen/). This enables viewing console output after disconnecting/when reconnecting to the client. 

For instance, do
```sh
screen -S pt
```
to start a named screen session. Then run a script, for instance
```
sh run_dim.sh
```
If you lose connection or want to quit the session, you can close your terminal. When you have started a new `ssh` connection later, you can do 
```
screen -r pt
```
to reconnect to the screen session.
