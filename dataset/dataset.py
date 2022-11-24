import torch
import transformers

import sys
sys.path.append('../')
import utils.util as util
import config.config as cfg

class MyDataset(torch.utils.data.Dataset):
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

        text_tokenized = self.tokenizer(text, padding='max_length', max_length=cfg.MAX_SIZE, truncation=True, return_tensors="pt")  # only for xxxTokenizerFast
        new_label = self.align_label(text_tokenized, label)
        new_label = torch.Tensor(new_label)
        
        return {
            'input_ids': text_tokenized.input_ids.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'token_type_ids': text_tokenized.token_type_ids.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'attention_mask': text_tokenized.attention_mask.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'target_tag': new_label.squeeze().type(torch.LongTensor).to(cfg.DEVICE)
        }
        # return text_tokenized.to(cfg.DEVICE), new_label.to(cfg.DEVICE)

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