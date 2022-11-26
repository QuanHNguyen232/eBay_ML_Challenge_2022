# eBay-ML-Challenge-2022 ([info](https://eval.ai/web/challenges/challenge-page/1733/overview))

Use `roberta_unfrz_epoch21_batch32_train.pt`

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

                                                                                                                                                                                                                                    | 0/125 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "D:\Coding-Workspace\eBay_ML_Challenge_2022\testing\train.py", line 43, in <module>
    train_loss = train_fn(train_dataloader, model, optimizer, scheduler)
  File "D:\Coding-Workspace\eBay_ML_Challenge_2022\testing\..\testing\engine.py", line 22, in train_fn
    out, loss = model(data['input_ids'], data['attention_mask'], data['token_type_ids'], data['target_tag'])
  File "C:\Users\QuanNguyen\anaconda3\envs\torch_py310\lib\site-packages\torch\nn\modules\module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "D:\Coding-Workspace\eBay_ML_Challenge_2022\testing\..\model\models.py", line 76, in forward
    loss = self.crf.neg_log_likelihood_loss(out, crf_mask, target_tags)
  File "D:\Coding-Workspace\eBay_ML_Challenge_2022\testing\..\model\nn.py", line 76, in neg_log_likelihood_loss
    numerator = self._compute_score(input, target, mask)
  File "D:\Coding-Workspace\eBay_ML_Challenge_2022\testing\..\model\nn.py", line 114, in _compute_score
    score += input[0, torch.arange(batch_size), target[0]]
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.