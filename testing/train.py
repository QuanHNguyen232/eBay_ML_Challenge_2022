import os
import torch
import gc
import numpy as np
from collections import defaultdict

import sys
sys.path.append('../')
import config.config as cfg
import model.models as models
from utils.util import get_data
from model.metrics import get_optimizer_scheduler
from dataset.dataset import MyDataset
from testing.engine import train_fn, eval_fn

sentences, labels, tags, label2id, id2label = get_data()
num_tags = len(list(cfg.encode_tag.classes_))

train_sentences, test_sentences = sentences[:cfg.split_pt], sentences[cfg.split_pt:]
train_labels, test_labels = labels[:cfg.split_pt], labels[cfg.split_pt:]
train_tags, test_tags = tags[:cfg.split_pt], tags[cfg.split_pt:]
print('data created')

train_dataset = MyDataset(train_sentences, train_labels)
valid_dataset = MyDataset(test_sentences, test_labels)
print('dataset created')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)
print('dataloader created')

# model = models.BERTModel(num_tags, isFreeze=True)
# model = models.RoBERTa_BiLSTM_CRF_Model(num_tags)
model = models.RoBERTa_BiLSTM_Model(num_tags, isFreeze=True)
model.load_state_dict(torch.load(os.path.join(cfg.SAVED_MODEL_DIR, 'roberta-bilstm_frz_batch16_lr0.0001_epo198_best.pt')))
model = model.to(cfg.DEVICE)
print('model created')

optimizer, scheduler = get_optimizer_scheduler(model, train_sentences)
# optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
print('optimizer & scheduler created')

best_train_loss = np.inf
best_valid_loss = np.inf
history = defaultdict(list)
isFrz = 'frz' if model.isFreeze else 'unfrz'
for epoch in range(cfg.EPOCHS):
    train_loss = train_fn(train_dataloader, model, optimizer, scheduler=None)
    valid_loss = eval_fn(valid_dataloader, model)
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)

    print(f"EPOCH = {epoch} \t train_loss = {train_loss} \t valid_loss = {valid_loss}")
    
    filename = f'{model.model_name}_{isFrz}_epo{epoch}_batch{cfg.BATCH_SIZE}_lr{str(cfg.LR)}'
    if train_loss < best_train_loss or valid_loss < best_valid_loss:
        torch.save(model.state_dict(), os.path.join(cfg.SAVED_MODEL_DIR, f'{filename}.pt'))
    
    best_train_loss = train_loss if train_loss < best_train_loss else best_train_loss
    best_valid_loss = valid_loss if valid_loss < best_valid_loss else best_valid_loss

gc.collect()    # garbage collector consumes lots of time
torch.cuda.empty_cache()

# save_train_log(model, history)