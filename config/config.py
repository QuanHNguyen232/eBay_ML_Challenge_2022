import torch
from sklearn.preprocessing import LabelEncoder


DATA_DIR = '../data/eBay_ML_Challenge_Dataset_2022'
TRAIN_DATA = 'Train_Tagged_Titles.tsv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SIZE = 64
SAVED_MODEL_DIR = '../saved/models'

EPOCHS = 50
BATCH_SIZE = 32

encode_tag = LabelEncoder()

class TOKENIZER:
    Token_BERT_ver = 'bert-base-uncased'
    label_all_tokens = True

class MODEL:
    BERT_ver = 'bert-base-uncased'