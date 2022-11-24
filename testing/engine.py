import gc
import torch

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm

import sys
sys.path.append('../')
import utils.util as util
import config.config as cfg



def train_fn(data_loader, model, optimizer, scheduler):
    model = model.to(cfg.DEVICE)
    model.train()
    total_loss = 0.0
  
    for data in tqdm(data_loader):
        # optimizer.zero_grad()
        out, loss = model(data['input_ids'], data['attention_mask'], data['token_type_ids'], data['target_tag'])
        
        loss.backward()
        # optimizer.step()
        # scheduler.step()

        total_loss += loss.item()
        
        del data
        gc.collect()
        torch.cuda.empty_cache()
    
    return total_loss / len(data_loader)