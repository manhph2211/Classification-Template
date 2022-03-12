IR-Project
====

# Introduction

- This project aims to classifying 9 in-bed posture using different methods ...

- Clone the project by `git clone https://github.com/manhph2211/IR-Project.git`

# Packages

```
comet_ml
opencv-python
torch
torchvision
```

# Data

- Here we want to test our method with 5-Fold Cross-Validation. 

- We just need to put `FOLD_i` into folder `./data`

# Usage

- First, set the value of `ROOT` in config file so that you can train the case(Fold_i) you want.

- Then, for experimental logging info, I used framework `comet_ml`. We need to create a file called `experiment_apikey.txt`. This file will just contain your api_key that the main website of `comet_ml` provides you when you create your own account.

- For create data form for dataset, we need config files. Here just run `python3 utils.py`

- For training, we run `python3 main.py`