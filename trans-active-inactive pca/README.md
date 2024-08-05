
# Overview

A small sample learning in Cas12a protein collateral trans-cleavage activity classification analysis with Principal Component Analysis (PCA)

## Requirements
* autogluon==1.1.1
* numpy==1.24.3
* pandas==2.0.3
* scikit-learn==1.3.0

While we have not tested with other versions, any reasonably recent versions of these requirements should work.

## General usage

### Data Preparation

Put 'train.csv' and 'test.csv' in the same folder with `main.py`. 

### train / predict / output

```
$ python main.py

```

An output file will be created in the same folder:

![Output Sample](https://github.com/LUCA-BioTech/cas_classification/blob/main/trans-active-inactive%20pca/output_sample.jpg)