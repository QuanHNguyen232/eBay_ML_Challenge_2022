import os
import torch
import numpy as np
import transformers

import sys
sys.path.append('../')
import config.config as cfg
import model.models as models
from utils.util import get_data
from dataset.dataset import MyDataset

sentences, labels, tags, label2id, id2label = get_data()
num_tags = len(list(cfg.encode_tag.classes_))

idx = 4001
sent = sentences[idx]
label = labels[idx]
tag = tags[idx]


dataset = MyDataset([sent], [label])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE)

for data in dataloader:
  print(data['input_ids'].shape, data['token_type_ids'].shape, data['attention_mask'].shape, data['target_tag'].shape)
  break

model = models.BERTModel(num_tags)
model.load_state_dict(torch.load(os.path.join(cfg.SAVED_MODEL_DIR, 'bert_epoch49_bacth32_best_valid.pt')))
model = model.to(cfg.DEVICE)


with torch.no_grad():
    for data in dataloader:
        out, loss = model(data['input_ids'],
                        data['attention_mask'],
                        data['token_type_ids'],
                        data['target_tag'])

        print(out.shape)
        print(cfg.encode_tag.inverse_transform(out.argmax(2).cpu().numpy().reshape(-1)))
        print(sent)