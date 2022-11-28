# eBay-ML-Challenge-2022 ([info](https://eval.ai/web/challenges/challenge-page/1733/overview))

Team *qua_n_lity__23_v2*



Best (trained on 4500 sentences + batch 32):
1. Roberta: (sum=66epoches(lr0.0001) + 28epoches(lr0.0003) + 23epoches(lr0.001 | train_loss = 2.143281674554162 | valid_loss = 1.8781160786747932)
    * `roberta_frz_epo22_batch32_lr0.001_valid.pt`: f1= 0.27955444513192473 (submit3.tsv)

1. Roberta+BiLSTM: (sum=51epoches(lr0.0003) + 27epoches(lr=0.0001) + 23(lr=3e-5) + 43(lr=0.001) + 100(lr=0.0001) | train_loss = 1.4 | valid_loss = 1.2):
    * `roberta-bilstm_frz_batch16_lr0.0001_epo100_best.pt`: f1=	0.45830829939093515 (submit1.tsv)
    * `roberta-bilstm_frz_batch16_lr0.0001_best.pt`: f1= 0.4389868151946224 (submit2.tsv)

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
