import torch
import transformers
from sklearn.preprocessing import LabelEncoder


DATA_DIR = '../data/eBay_ML_Challenge_Dataset_2022'
TRAIN_DATA = 'Train_Tagged_Titles.tsv'
TITLE_DATA = 'Listing_Titles.tsv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVED_MODEL_DIR = '../saved/models'
SAVED_SUBMIT_DIR = '../saved/submissions'

EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-4
num_warmup_steps = 0

encode_tag = LabelEncoder()
split_pt = 4500

class TOKENIZER:
    Token_BERT_ver = 'bert-base-uncased'
    Token_RoBERTa_ver = 'Jean-Baptiste/roberta-large-ner-english'
    label_all_tokens = True
    MAX_SIZE = 32
    tokenizer = transformers.BertTokenizerFast.from_pretrained(Token_BERT_ver)
    quiz_tkzer = transformers.BertTokenizerFast.from_pretrained(Token_BERT_ver)

class MODEL:
    BERT_ver = 'bert-base-uncased'
    RoBERTa_ver = 'Jean-Baptiste/roberta-large-ner-english'

train_log_path = '../saved/logs'