import torch
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append('../')
import config.config as cfg

def get_optimizer_scheduler(model, sentences):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    
    optimizer = AdamW(optimizer_parameters, lr=cfg.LR)
    
    num_train_steps = int(len(sentences) / cfg.BATCH_SIZE * cfg.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps)

    return optimizer, scheduler
