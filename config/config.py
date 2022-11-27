import torch
from sklearn.preprocessing import LabelEncoder


DATA_DIR = '../data/eBay_ML_Challenge_Dataset_2022'
TRAIN_DATA = 'Train_Tagged_Titles.tsv'
DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
SAVED_MODEL_DIR = '../saved/models'

EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
num_warmup_steps = 0

encode_tag = LabelEncoder()
split_pt = 4500

class TOKENIZER:
    Token_BERT_ver = 'bert-base-uncased'
    Token_RoBERTa_ver = 'Jean-Baptiste/roberta-large-ner-english'
    label_all_tokens = True
    MAX_SIZE = 32

class MODEL:
    BERT_ver = 'bert-base-uncased'
    RoBERTa_ver = 'Jean-Baptiste/roberta-large-ner-english'