#%%
import os
import gc
import re
import csv
import string
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict, Counter

import sys
sys.path.append('../')
import config.config as cfg
import model.models as models
from utils.util import get_data, get_quiz
from dataset.dataset import MyDataset, QuizDataset

torch.cuda.is_available()

#%%
_ = get_data()  # to init cfg.encode_tag, etc.
num_tags = len(list(cfg.encode_tag.classes_))
sent_ids, raw_sents, sentences, labels = get_quiz()
print('data created')

#%%
tokenizer = transformers.BertTokenizerFast.from_pretrained(cfg.TOKENIZER.Token_BERT_ver)
print('tkzer created')

#%%
# model = models.RoBERTa_BiLSTM_Model(num_tags, isFreeze=True)
model = models.BERTModel(num_tags)
model.load_state_dict(torch.load(os.path.join(cfg.SAVED_MODEL_DIR, 'roberta_frz_epo22_batch32_lr0.001_valid.pt')))
model = model.to(cfg.DEVICE)
print('model created')

#%%

return_sent_id = []
return_sent = []
return_tag = []
with torch.no_grad():
    model.eval()
    for index, (sent_id, raw_sent, sent) in tqdm(enumerate(zip(sent_ids, raw_sents, sentences)), total=len(raw_sents)):
        txt_tkzed = tokenizer(' '.join(sent), padding='max_length', max_length=cfg.TOKENIZER.MAX_SIZE, truncation=True, return_tensors="pt")
        data = {
            'sent_id': int(sent_id),
            'text_tokenized': txt_tkzed,
            'input_ids': txt_tkzed.input_ids.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'token_type_ids': txt_tkzed.token_type_ids.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
            'attention_mask': txt_tkzed.attention_mask.squeeze().type(torch.LongTensor).to(cfg.DEVICE),
        }

        pred = model(data['input_ids'].unsqueeze(0),
                    data['attention_mask'].unsqueeze(0),
                    data['token_type_ids'].unsqueeze(0))    # out.shape=(1, 32, 32) as (batch,seq_len,label)
        
        word_ids = txt_tkzed.word_ids()
        out = pred.argmax(2).detach().cpu().numpy().reshape(-1)
        conf = pred.detach().cpu().numpy().max(2).reshape(-1)
        words = cfg.TOKENIZER.tokenizer.decode(data['input_ids'])
        
        result = defaultdict(list)
        for i, (word_id, o, c) in enumerate(zip(word_ids, out, conf)):
            if word_id is None: continue
            result[word_id].append((cfg.encode_tag.inverse_transform([o])[0], c))
        
        for key in result.keys():
            result[key].sort(key=lambda x: x[1], reverse=True)
        
        last_word_id = max(result.keys())
        start_idx = list(data['input_ids'].detach().cpu().numpy().reshape(-1)).index(101)
        end_idx = list(data['input_ids'].detach().cpu().numpy().reshape(-1)).index(102)
        
        
        fin_words = words.split(' ')[start_idx+1 : last_word_id + 1 + 1]    # +2 'cause 1st idx [CLS] (101) is removed, and include last_word_id
        keys_sort = sorted(result.keys())
        
        # process result: get most predicted tag: e.g. 2 brand + 1 Type => Brand
        for key in keys_sort:
            val = Counter(result[key])

            res = ''
            max_count = 0
            for label in result[key]:
                if max_count<val[label]:
                    res = label
                    max_count = val[label]
            
            result[key]=res
        
        # MATCHING WORD-TAG
        tmp_dict2 = defaultdict(list)
        for word, key in zip(fin_words, keys_sort):
            tmp_dict2[word].append(result[key][0])

        ret = []
        for word in raw_sent:
            word = word.lower()
            if len(word)==1 and word in string.punctuation:
                ret.append('No Tag')    # punctuations
            elif word == '\'s':
                ret.append(ret[-1] if len(ret)>0 else 'No Tag') # 's: get tag from prev
            else:
                tmp_word = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", word)
                if tmp_word in tmp_dict2.keys():
                    ret.append(tmp_dict2[tmp_word].pop(0))
                else:
                    ret.append('No Tag')
        
        return_sent_id = return_sent_id + [sent_id for _ in raw_sent]
        return_sent = return_sent + raw_sent
        return_tag = return_tag + ret

        
        if len(ret) != len(raw_sent):
            print('ERROR', 'len(ret) != len(raw_sent)')
            break
        
        if (len(return_sent_id) != len(return_tag)) or (len(return_tag) != len(return_sent)):
            print('ERROR', 'len(return_tag) != len(return_sent) OR len(return_sent_id) != len(return_tag)')
            break


gc.collect()
torch.cuda.empty_cache()

df = pd.DataFrame(data={
                    'Record Number': return_sent_id,
                    'Aspect Name': return_tag,
                    'Aspect Value': return_sent
                })
df.to_csv(os.path.join(cfg.SAVED_SUBMIT_DIR, 'submit3.tsv'), sep='\t' , header=False, index=False, quoting=csv.QUOTE_NONE)

#%% EVALUATION

# names = [f'roberta_unfrz_epoch{i}_batch32_train' for i in range(5, 21)]
# for name in names:
#     model = models.BERTModel(num_tags)
    # model.load_state_dict(torch.load(os.path.join(cfg.SAVED_MODEL_DIR, f'{name}.pt')))
    # model = model.to(cfg.DEVICE)
    # model.eval()

    # sum_f1_macro = 0.0
    # sum_f1_micro = 0.0
    # with torch.no_grad():
    #     for data in dataloader:
    #         out, loss = model(data['input_ids'],
    #                         data['attention_mask'],
    #                         data['token_type_ids'],
    #                         data['target_tag'])
            
#             # print(data['input_ids'].shape,
#             #         data['attention_mask'].shape,
#             #         data['target_tag'].shape,
#             #         out.shape)   # [1, 64]

#             mask = data['target_tag'].view(-1) != -100
#             active_gt = torch.where(
#                 mask,
#                 data['target_tag'].view(-1),
#                 torch.tensor(cfg.encode_tag.transform(np.array(['No Tag']))[0]).type_as(data['target_tag'])
#             )

#             gt = active_gt.cpu().numpy().reshape(-1)
#             pred = out.argmax(2).cpu().numpy().reshape(-1)

#             gt_tag = cfg.encode_tag.inverse_transform(gt)
#             pred_tag = cfg.encode_tag.inverse_transform(pred)
            
#             y_true = []
#             y_pred = []
#             for y, y_hat, attention in zip(gt, pred, data['attention_mask'].view(-1)):
#                 if attention == 1:
#                     y_true.append(y)
#                     y_pred.append(y_hat)
            
#             y_true, y_pred = np.array(y_true), np.array(y_pred)
#             sum_f1_macro += f1_score(y_true, y_pred, average='macro')
#             sum_f1_micro += f1_score(y_true, y_pred, average='micro')

#         print(name)
#         print(f'avg f1_macro: {sum_f1_macro/len(dataloader)}')
#         print(f'avg f1_micro: {sum_f1_micro/len(dataloader)}')
#         print()

