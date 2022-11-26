import os
import torch
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

split_pt = 4000
train_sentences, test_sentences = sentences[:split_pt], sentences[split_pt:]
train_labels, test_labels = labels[:split_pt], labels[split_pt:]
train_tags, test_tags = tags[:split_pt], tags[split_pt:]
print('data created')

train_dataset = MyDataset(train_sentences, train_labels)
valid_dataset = MyDataset(test_sentences, test_labels)
print('dataset created')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)
print('dataloader created')

model = models.BERTModel(num_tags)
model = model.to(cfg.DEVICE)
print('model created')

optimizer, scheduler = get_optimizer_scheduler(model, train_sentences)
print('optimizer & scheduler created')

best_train_loss = np.inf
best_valid_loss = np.inf
history = defaultdict(list)

for epoch in range(cfg.EPOCHS):
    train_loss = train_fn(train_dataloader, model, optimizer, scheduler)
    valid_loss = eval_fn(valid_dataloader, model)
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)

    print(f"EPOCH = {epoch} \t train_loss = {train_loss} \t valid_loss = {valid_loss}")

    if train_loss < best_train_loss:
        torch.save(model.state_dict(), os.path.join(cfg.SAVED_MODEL_DIR, f'{model.model_name}_epoch{epoch}_bacth{cfg.BATCH_SIZE}_best_train.pt'))
        best_train_loss = train_loss
    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), os.path.join(cfg.SAVED_MODEL_DIR, f'{model.model_name}_epoch{epoch}_bacth{cfg.BATCH_SIZE}_best_valid.pt'))
        best_valid_loss = valid_loss