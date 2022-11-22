#%%
import csv
import os
import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import config.config as cfg

#%%
file_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_DATA)
df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
df


#%%
df = df.fillna(method="ffill")
df

#%% Encode label
df['Tag'] = cfg.encode_tag.fit_transform(df['Tag'])
df

#%% Convert df to np array
sentences = df.groupby("Record Number")["Token"].apply(list).values
tags = df.groupby("Record Number")["Tag"].apply(list).values
sentences[0], tags[0]

#%%
''' Text pre-processing
'''
for i, (sentence, tag) in enumerate(zip(sentences, tags)):
    # word level: remove punctuation
    remove_i = []
    for idx, _ in enumerate(sentence):
        sentence[idx] = sentence[idx].lower()
        if sentence[idx] in string.punctuation+'--': remove_i.append(idx)
    sentences[i] = np.delete(sentence, remove_i)
    tags[i] = np.delete(tag, remove_i)

    # word level (multi-color -> multicolor)
    for idx, _ in enumerate(sentences[i]):
        sentences[i][idx] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", sentences[i][idx])
    
    
    
#%% PLOTS
df["Tag"].value_counts().plot(kind="bar", figsize=(10,5));

# We also want to check the length of the sentences in the dataset because we need to decide how to set up the max length in our model later.
word_counts = df.groupby("Record Number")["Token"].agg(["count"])
word_counts.hist(bins=50, figsize=(8,6));

