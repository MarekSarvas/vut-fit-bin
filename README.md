# VUT FIT BIN - Neuroevolution
**Author**: Marek Sarvas \
**Login**: xsarva00

## Run
1. install enviroment with dependencies in **requirements.txt** with *install.sh* or by running 
```
make setup
```
2. download data if needed by running 
```
make dataset
```
3. change experiment parameters in *run.sh* and run experiments with 
```
bash run.sh 2 2
```
or 
```
make
```
4. to create plots and *.tex* tables of each experiment run
```
make plots
```
or run.sh with stage 3
## Implementation
- This project  aims to implement paper by Lingxi Xie and Alan Yuille (https://arxiv.org/pdf/1703.01513.pdf).
- part of evolution algorithm and chromosome implementation are in **src/** folder
- **src/models/** contains some small basic feed-forward convolutional baseline models trained on small number of epochs 