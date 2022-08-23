IR-Project
====

# Introduction :smile:

In this project, I provided an simple end-to-end implemetation for training, evaluating and monitoring a stardard classifier which is considered as a common template and easy-to-use for AI newbies. I did apply several techniques such as Expotential Moving Average (EMA), Label Smoothing, Modern Optimizer, K-Fold cross-validation, random-augmentations ... traking logs, hyperparameters ... with comet ML tool.

# Packages

```
comet_ml
timm
opencv-python
torch
torchvision
```

# Data

Here we want to test our method with 5-Fold Cross-Validation. We just need to put `FOLD_i` into folder `./data`.

# Usage

First, set the value of `ROOT` in config file `config.yml` so that you can train the case(Fold_i) you want.

Then, for experimental logging info, I used framework `comet_ml`. We need to create a file called `experiment_apikey.txt`. This file will just contain your api_key that the main website of `comet_ml` provides you when you create your own account.

For create data form for dataset, we need config files. Here just run `python3 utils.py`

For training, we run `python3 main.py`
