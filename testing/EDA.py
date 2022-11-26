#%%
import csv
import os
import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers

import sys
sys.path.append('../')
import config.config as cfg

#%%
file_path = os.path.join(cfg.DATA_DIR, cfg.TRAIN_DATA)
df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
df = df.fillna(method="ffill")
df['Label'] = cfg.encode_tag.fit_transform(df['Tag'])

#%%
num_labels = len(df["Tag"].value_counts()) if len(df["Tag"].value_counts()) == len(df["Label"].value_counts()) else -100
if num_labels==-100: raise Exception("Sorry, df[\"Tag\"].value_counts()) == len(df[\"Label\"].value_counts() returns FALSE")

#%% Convert df to np array
sentences = df.groupby("Record Number")["Token"].apply(list).values
labels = df.groupby("Record Number")["Label"].apply(list).values
tags = df.groupby("Record Number")["Tag"].apply(list).values
sentences[0], labels[0], tags[0]





#%% PLOTS
df["Tag"].value_counts().plot(kind="bar", figsize=(10,5));

# We also want to check the length of the sentences in the dataset because we need to decide how to set up the max length in our model later.
word_counts = df.groupby("Record Number")["Token"].agg(["count"])
word_counts.hist(bins=50, figsize=(8,6));

#%%
def test_file():
    print('file test/EDA.py')


#%%
model = transformers.BertModel.from_pretrained('Jean-Baptiste/roberta-large-ner-english')
model