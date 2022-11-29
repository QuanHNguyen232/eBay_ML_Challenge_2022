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
from dataset import dataset
import utils.util as util

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


#%% #########################
########### RUN TEST ########
#############################
#%%
sent_ids, sentences, labels = util.get_quiz()
_ = util.get_data()
#%%
sent_ids[0], sentences[0]
#%%
train_dataset = dataset.QuizDataset(sent_ids, sentences)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)

#%%
data = train_dataset[9]
data['sent_id'], data['sent_id'].shape, data['input_ids'].shape

#%%
for data in trainloader:
    print(data['input_ids'].shape)
    break
#%%
