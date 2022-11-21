#%%
import csv
import os
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

#%%
df["Tag"].value_counts().plot(kind="bar", figsize=(10,5));

#%%
# We also want to check the length of the sentences in the dataset because we need to decide how to set up the max length in our model later.
word_counts = df.groupby("Record Number")["Token"].agg(["count"])
word_counts.hist(bins=50, figsize=(8,6));