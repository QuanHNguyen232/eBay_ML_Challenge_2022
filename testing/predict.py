import os
import torch
import numpy as np
import transformers
from tqdm import tqdm
from sklearn.metrics import f1_score

import sys
sys.path.append('../')
import config.config as cfg
import model.models as models
from utils.util import get_data
from dataset.dataset import MyDataset

sentences, labels, tags, label2id, id2label = get_data()
num_tags = len(list(cfg.encode_tag.classes_))

sent = sentences[cfg.split_pt:]
label = labels[cfg.split_pt:]
tag = tags[cfg.split_pt:]


dataset = MyDataset(sent, label)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE)

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
        
model = models.RoBERTa_BiLSTM_CRF_Model(num_tags)
model.load_state_dict(torch.load(os.path.join(cfg.SAVED_MODEL_DIR, 'roberta-bilstm-crf_frz_epoch36_batch32_valid.pt')))
model = model.to(cfg.DEVICE)
model.eval()
with torch.no_grad():
    for data in dataloader:
        out, loss = model(data['input_ids'], data['attention_mask'], data['token_type_ids'], data['target_tag'])
        # out = torch.tensor(out)
        idx = 24
        lim = len(out[idx])
        print('target',cfg.encode_tag.inverse_transform(data['target_tag'][idx][:lim].detach().cpu().numpy()))
        print('predic',out[idx])
        print('target',data['target_tag'][idx][:lim].detach().cpu().numpy())
        print(dataset.tokenizer.convert_ids_to_tokens(data['input_ids'][idx][:lim].detach().cpu().numpy()))
        print(dataset.tokenizer.decode(data['input_ids'][idx][:lim].detach().cpu().numpy()))
        break