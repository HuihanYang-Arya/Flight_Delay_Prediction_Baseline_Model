# Baseline Models for Flight Delay Prediction

Author: Huihan Yang

Some code sources: https://github.com/Kaimaoge/STPN.git

Note: This repo is still under MAINTAIN, some code might be ugly.

## Data

Our data contains Chinese flight delay data and USA flight delay data. 

### U.S.

* U.S. data can be found in `udata`. To use it, please change the parameter data to "US" in `configs.json`.

U.S. dataset is collected from the BTS database from 2015-2021. We select 70 airports with heavier traffic volumes for experiments. 

### China

* Chinese data can be found in `cdata`. To use it, please change the parameter data to "China" in `configs.json`.

Chinese dataset is collected from Xiecheng from 2015-2017.

Note that for both dataset, only flights between 6am and 12 pm are considered.

## Baseline Model 

1. Historical Average (HA)

2. Variational Auto Regression (VAR)

3. Auto Regression Indifference Moving Average (ARIMA)

4. Support Vector Machine (SVM)

For the above four methods, the implementation is in `baseline_methods.py`. To see the result, you could run in `baseline_model.ipynb`

5. Long Short Term Memory (LSTM) 

6. Gated Recurrent Unit (GRU)

* This one may have some bugs, under MAINTAINACE

7. Graph Attention Network (GAT)

8. Attention Based Spatial-Temporal Graph Convolutional Network (ASTGCN)

* https://github.com/guoshnBJTU/ASTGCN-r-pytorch.git

For the above four methods, the implementation is in `model-name.py`. To see the result, you should modify the parameter in the `configs.json` first. Please note that for LSTM, the batch size should be set as 1. Then you could run by `python train_model_name.py`.

One possible retults to compare are in `baseline_results_possible.png`.

### Models in folder `GNN_baseline`

Contains models that has not been trained results. 


