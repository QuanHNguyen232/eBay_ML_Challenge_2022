import torch
import transformers

import sys
sys.path.append('../')
from utils.util import get_data
import config.config as cfg

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, label_all_tokens=cfg.TOKENIZER.label_all_tokens):
        self.sentences = sentences
        self.labels = labels
        self.label_all_tokens = label_all_tokens

        self.tokenizer = cfg.TOKENIZER.tokenizer
    
    def __len__(self):
        return len(self.sentences) if len(self.sentences)==len(self.labels) else -100
    
    def __getitem__(self, index):
        text = ' '.join(self.sentences[index])
        label = self.labels[index]

        text_tokenized = self.tokenizer(text, padding='max_length', max_length=cfg.TOKENIZER.MAX_SIZE, truncation=True, return_tensors="pt")  # only for xxxTokenizerFast
        new_label = self.align_label(text_tokenized, label)
        new_label = torch.Tensor(new_label)
        
        return {
            'text_tokenized': text_tokenized,
            'input_ids': text_tokenized.input_ids.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'token_type_ids': text_tokenized.token_type_ids.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'attention_mask': text_tokenized.attention_mask.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'target_tag': new_label.squeeze().type(torch.LongTensor).to(cfg.DEVICE)
        }

    def align_label(self, text_tokenized, labels):
        word_ids = text_tokenized.word_ids()    # only for xxxTokenizerFast
        previous_word_idx = None
        label_ids = []
        # special_tk_tag = list(cfg.encode_tag.classes_).index('No Tag')
        special_tk_tag = -100
        
        for word_idx in word_ids:
            if word_idx is None:  # Set the special tokens to -100.
                label_ids.append(special_tk_tag)
            elif word_idx != previous_word_idx: # Only label the first token of a given word.
                try:
                    label_ids.append(labels[word_idx])
                except:
                    label_ids.append(special_tk_tag)
            else:
                label_ids.append(labels[word_idx] if self.label_all_tokens else special_tk_tag)
            
            previous_word_idx = word_idx
        
        return label_ids

class QuizDataset(torch.utils.data.Dataset):
    def __init__(self, sent_ids, sentences, label_all_tokens=cfg.TOKENIZER.label_all_tokens):
        self.sent_ids = sent_ids
        self.sentences = sentences
        self.label_all_tokens = label_all_tokens

        self.quiz_tkzer = cfg.TOKENIZER.quiz_tkzer
    
    def __len__(self):
        return len(self.sentences) if len(self.sentences)==len(self.sent_ids) else -100
    
    def __getitem__(self, index):
        sent_id = self.sent_ids[index]
        text = ' '.join(self.sentences[index])
        
        text_tkzed = self.quiz_tkzer(text, padding='max_length', max_length=cfg.TOKENIZER.MAX_SIZE, truncation=True, return_tensors="pt")
        
        return {
            'sent_id': sent_id,
            'text_tokenized': text_tkzed,
            'input_ids': text_tkzed.input_ids.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'token_type_ids': text_tkzed.token_type_ids.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'attention_mask': text_tkzed.attention_mask.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
        }
