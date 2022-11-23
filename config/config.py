import torch
from sklearn.preprocessing import LabelEncoder


DATA_DIR = '../data/eBay_ML_Challenge_Dataset_2022'
TRAIN_DATA = 'Train_Tagged_Titles.tsv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

encode_tag = LabelEncoder()

class TOKENIZER:
    Token_BERT_ver = 'bert-base-uncased'
    label_all_tokens = True