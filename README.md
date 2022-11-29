# eBay-ML-Challenge-2022 ([info](https://eval.ai/web/challenges/challenge-page/1733/overview))

Team *qua_n_lity__23_v2*

**Note**:
> LoL, I have no idea why RoBERTa-BiLSTM and RoBERTa (or BERTModel in models) have worse performance than BERTModel_1, which is a simple model. It was trained on just around 50epochs while the other 2 were trained for >100 epochs (based on f1-score w/ scikit-learn) and their loss (CrossEntropy) are lower, too. 

> What caused that? *My guess: maybe those 2 are too big for this small amount of data???*
> * a small change: special_tk_tag = -100 instead of 19 (or 'No Tag') from `dataset`

Best (trained on 4500 sentences + batch 32):
1. BERT: `bert_epoch2_bacth32_best_valid.pt`: f1 = 0.7227390454084825
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
