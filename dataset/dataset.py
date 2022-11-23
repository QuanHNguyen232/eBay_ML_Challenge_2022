import torch
import transformers

import sys
sys.path.append('../')
import utils.util as util
import config.config as cfg

class MyData(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, label_all_tokens=cfg.TOKENIZER.label_all_tokens):
        self.sentences = sentences
        self.labels = labels
        self.label_all_tokens = label_all_tokens

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(cfg.TOKENIZER.Token_BERT_ver)
    
    def __len__(self):
        return len(self.sentences) if len(self.sentences)==len(self.labels) else -100
    
    def __getitem__(self, index):
        text = ' '.join(self.sentences[index])
        label = self.labels[index]

        text_tokenized = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")  # only for xxxTokenizerFast
        new_label = self.align_label(text_tokenized, label)
        new_label = torch.Tensor(new_label)
        return text_tokenized.to(cfg.DEVICE), new_label.to(cfg.DEVICE)

    def align_label(self, text_tokenized, labels):
        word_ids = text_tokenized.word_ids()    # only for xxxTokenizerFast
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:  # Set the special tokens to -100.
                label_ids.append(-100)
            elif word_idx != previous_word_idx: # Only label the first token of a given word.
                try:
                    label_ids.append(labels[word_idx])
                except:
                    label_ids.append(-100)
            else:
                label_ids.append(labels[word_idx] if self.label_all_tokens else -100)
            
            previous_word_idx = word_idx
        
        return label_ids


#### UNIT TEST ###
# data = util.get_data()
# print(data['num_labels'], len(cfg.encode_tag.classes_))
# sentences, labels = data['sentences'], data['labels']
# dataset = MyData(sentences, labels)
# print('len dataset',len(dataset))

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
# for data, label in dataloader:
#   break

# print(data.input_ids.shape, data.token_type_ids.shape, data.attention_mask.shape, label.shape)
# print(type(data.input_ids), type(data.token_type_ids), type(data.attention_mask), type(label))
# print(data.input_ids.device, data.token_type_ids.device, data.attention_mask.device, label.device)