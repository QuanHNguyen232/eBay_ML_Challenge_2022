import os
import csv
import string
import re
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
import config.config as cfg

def load_data():
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

    return sentences, labels, tags, label2id, id2label

def preprocess(sentences, labels, tags, isUnCase=True, rm_punc=True, rm_word_symbol=True):
    # isCase: Case sentitive (english != English): "roberta-large"
    # isUnCase: bert-base-uncased or Jean-Baptiste/roberta-large-ner-english

    for i, (sentence, label, tag) in enumerate(zip(sentences, labels, tags)):
        # word level: remove punctuation
        remove_i = []
        for idx, _ in enumerate(sentence):
            if isUnCase: sentence[idx] = sentence[idx].lower()
            if rm_punc and sentence[idx] in string.punctuation+'--': remove_i.append(idx)
        
        sentences[i] = np.delete(sentence, remove_i)
        labels[i] = np.delete(label, remove_i)
        tags[i] = np.delete(tag, remove_i)

        # word level (multi-color -> multicolor)
        if rm_word_symbol:
            for idx, _ in enumerate(sentences[i]):
                sentences[i][idx] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", sentences[i][idx])
    
    return sentences, labels, tags

def get_data():
    sentences, labels, tags, label2id, id2label = load_data()
    sentences, labels, tags = preprocess(sentences, labels, tags)

    return sentences, labels, tags, label2id, id2label


def emoji_remove(text):
    ''' use for inference (run on eval mode)

    NEED CHECK IF '-' RETURNS 'no-tag'
    '''
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", re.UNICODE)
    return emoji_pattern.sub(r'-', text)