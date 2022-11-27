# eBay-ML-Challenge-2022 ([info](https://eval.ai/web/challenges/challenge-page/1733/overview))

Use `roberta_unfrz_epoch21_batch32_train.pt`
Best (trained on 4500 sentences + batch 32):
1. Roberta: (sum=66epoches(lr0.0001) + 28epoches(lr0.0003) + 23epoches(lr0.001 | train_loss = 2.143281674554162 | valid_loss = 1.8781160786747932)
        * `roberta_frz_epo22_batch32_lr0.001_valid.pt`
1. Roberta+BiLSTM: (sum=51epoches(lr0.0003) + 27epoches(lr=0.0001) + 23(lr=3e-5) + 43(lr=0.001) | train_loss = 1.5407367982762925 | valid_loss = 1.3700652867555618):
        * `roberta-bilstm_frz_epo42_batch16_lr0.0001.pt`
        * `roberta-bilstm_frz_epo26_batch32_lr0.0001.pt`
        * `roberta-bilstm_frz_epo21_batch32_lr0.0003_train.pt`
        * `roberta-bilstm_frz_epo28_batch32_lr0.0003_train.pt`

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
