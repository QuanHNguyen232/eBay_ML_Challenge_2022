# eBay-ML-Challenge-2022

Link: [https://eval.ai/web/challenges/challenge-page/1733/overview](https://eval.ai/web/challenges/challenge-page/1733/overview)

Src (11/1/2022):
1. https://pytorch.org/hub/pytorch_fairseq_roberta/ (RoBERTa)
    * https://github.com/facebookresearch/fairseq
        * https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md
        * https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md
1. Huggingface-Transformers (RoBERTa):
    * https://github.com/huggingface/transformers/tree/77321481247787c97568c3b9f64b19e22351bab8
    * https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/roberta#transformers.RobertaForTokenClassification
1. DeBERTa:
    * https://github.com/microsoft/DeBERTa
1. Paper:
    * https://arxiv.org/abs/1907.11692
1. NER with BERT:
    * https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
1. NER with RoBERTa:
    * https://www.kaggle.com/code/eriknovak/pytorch-roberta-named-entity-recognition
    * https://github.com/code2ashish/Name-Entity-Recognition-BERT-distilBERT-RoBERTa

Note: Anh Thái: A thấy bài này thì pretrained model + CRF là ok r. Thi thì thêm ensemble nữa. ensemble pre trained model (như RoBERTa, DeBERTa,... này nọ)
1. https://towardsdatascience.com/named-entity-recognition-ner-with-bert-in-spark-nlp-874df20d1d77
1. https://github.com/Louis-udm/NER-BERT-CRF
1. https://medium.com/data-science-in-your-pocket/named-entity-recognition-ner-using-conditional-random-fields-in-nlp-3660df22e95c
1. https://towardsdatascience.com/conditional-random-field-tutorial-in-pytorch-ca0d04499463
1. https://jesusleal.io/2020/10/20/RoBERTA-Text-Classification/
1. https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/
1. https://huggingface.co/docs/transformers/model_doc/roberta
1. https://www.projectpro.io/article/bert-nlp-model-explained/558#mcetoc_1fr22q9rkf


---
## Folder Structure
```
.
├───config
│       config.py - configuration
│
├───data - default directory for storing input data
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