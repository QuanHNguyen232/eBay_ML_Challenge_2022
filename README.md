# eBay-ML-Challenge-2022 ([info](https://eval.ai/web/challenges/challenge-page/1733/overview))

pi---
## Folder Structure
```
.
├───config
│       config.py - configuration
│
├───data - default directory for storing input data
│   └───eBay_ML_Challenge_Dataset_2022 - all unzipped data in this folder (.tsv files)
│
├───data_loader - anything about data loading
│       data_loaders.py
│
├───model - this folder contains any net of project.
│       losses.py
│       metrics.py
│       models.py
│
├───saved
│   ├───logs - default logdir for tensorboard and logging
│   ├───models - trained models are saved here
│   └───submissions - submission file are saved here
│
├───test - test functions
│       a.py
│
├───tools - open source are saved here
│       RoBERTa.py
│
└───utils - small utility functions
        util.py
```
