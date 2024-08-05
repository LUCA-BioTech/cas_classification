
# Overview

A high-precision protein classification model

## Requirements
* accelerate==0.23.0
* deepspeed==0.10.3
* pandas==2.0.0
* python-box==7.0.1
* scikit-learn==1.2.2
* scipy==1.10.1
* torch==1.13.1+cu116
* wandb==0.15.2
* pyfaidx==0.6.0
* PyYAML==6.0

While we have not tested with other versions, any reasonably recent versions of these requirements should work.

## General usage

### Fetching data
Run `collect_dataset.py` to fetch the CRISPR-associated protein dataset. 

### train / eval / predict

```
$ python esm2_classification_simple.py

optional arguments:
    --config                        Path to the YAML config file
    --action                        "train", "eval", "predict"
    --output_dir                    output directory
    --output_file                   predict/export output file
    --output_loss_on_prediction     whether output loss on prediction
    --local_rank                    local rank
```

For example:
```
# train a model
python esm2_classification_simple.py -c config/cas-classification-train-15B.yml -a train -o output/cas-classification-15B

```