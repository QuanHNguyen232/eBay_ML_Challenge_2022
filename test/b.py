#%%
import sys
sys.path.append('../')
import csv
import os
import pandas as pd
from config.config import Config


#%%
file_path = os.path.join(Config.data_dir, Config.train_data)
df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
df
