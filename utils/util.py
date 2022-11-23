import os
import csv
import string
import re
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
import testing.EDA as eda
import config.config as cfg

def get_data(isUnCase=True):
    # isCase: Case sentitive (english != English): "roberta-large"
    # isUnCase: bert-base-uncased or Jean-Baptiste/roberta-large-ner-english

    file_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_DATA)
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
    df = df.fillna(method="ffill")
    df['Label'] = cfg.encode_tag.fit_transform(df['Tag'])

    label2id = {}
    id2label = {}
    for i, tag in enumerate(cfg.encode_tag.classes_):
        label2id[i] = tag
        id2label[tag] = i
    
    sentences = df.groupby("Record Number")["Token"].apply(list).values
    labels = df.groupby("Record Number")["Label"].apply(list).values
    tags = df.groupby("Record Number")["Tag"].apply(list).values

    for i, (sentence, label, tag) in enumerate(zip(sentences, labels, tags)):
        # word level: remove punctuation
        remove_i = []
        for idx, _ in enumerate(sentence):
            if isUnCase: sentence[idx] = sentence[idx].lower()
            if sentence[idx] in string.punctuation+'--': remove_i.append(idx)
        
        sentences[i] = np.delete(sentence, remove_i)
        labels[i] = np.delete(label, remove_i)
        tags[i] = np.delete(tag, remove_i)

        # word level (multi-color -> multicolor)
        for idx, _ in enumerate(sentences[i]):
            sentences[i][idx] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", sentences[i][idx])
    
    return {
        'sentences': sentences,
        'labels': labels,
        'tags': tags,
        'label2id': label2id,
        'id2label': id2label
    }
