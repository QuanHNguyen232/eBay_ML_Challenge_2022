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



def train_fn(data_loader, model, optimizer, scheduler=None):
    model.train()
    total_loss = 0.0
  
    for data in tqdm(data_loader):
        out, loss = model(data['input_ids'], data['attention_mask'], data['token_type_ids'], data['target_tag'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        total_loss += loss.item()
        
        del data
        gc.collect()
        torch.cuda.empty_cache()
    
    return total_loss / len(data_loader)

def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0.0
  
    with torch.no_grad():
        for data in tqdm(data_loader):
            out, loss = model(data['input_ids'], data['attention_mask'], data['token_type_ids'], data['target_tag'])
            
            total_loss += loss.item()
            
            del data
            gc.collect()
            torch.cuda.empty_cache()
    
    return total_loss / len(data_loader)