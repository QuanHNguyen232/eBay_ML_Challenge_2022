11/16/2022: Youtube tutorial
* [Fine Tuning BERT for Named Entity Recognition (NER)](https://www.youtube.com/watch?v=dzyDHMycx_c)
* [Train Custom NAMED ENTITY RECOGNITION (NER) model using BERT](https://www.youtube.com/watch?v=uKPBkendlxw)
* [How to Train a spaCy NER model (Named Entity Recognition for DH 04 | Part 03)](https://www.youtube.com/watch?v=7Z1imsp6g10&list=PL2VXyKi-KpYs1bSnT8bfMFyGS-wMcjesM&index=6)
* [Building an entity extraction model using BERT](https://www.youtube.com/watch?v=MqQ7rqRllIc)
* [Named Entity Recognition - Natural Language Processing With Python and NLTK p.7](https://www.youtube.com/watch?v=LFXsG7fueyk)


11/15/2022: may need both NER and POS
* NEED: read RoBERTa paper, learn how to train w/ custom labels
* Check Hugging Face's RoBERTa:
    * Basic tutorial: https://huggingface.co/course/chapter7/2?fw=pt
    * RoBERTa's docs: https://huggingface.co/docs/transformers/model_doc/roberta -> check RobertaForTokenClassification
* Check how to organize input/output:
    * https://towardsdatascience.com/named-entity-recognition-5324503d70da
    * https://newscatcherapi.com/blog/train-custom-named-entity-recognition-ner-model-with-spacy-v3

11/14/2022: Preprocessing NLP (cleaning)
* https://realpython.com/natural-language-processing-spacy-python/
* https://iq.opengenus.org/text-preprocessing-in-spacy/
* https://medium.com/voice-tech-podcast/implementing-a-simple-text-preprocessing-pipeline-with-spacy-597a3568bc19
* https://towardsdatascience.com/text-cleaning-for-nlp-in-python-2716be301d5d

11/13/2022: Check aio2022 record 11-12-2022 (basic tokenize)
* check spell, etc.

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