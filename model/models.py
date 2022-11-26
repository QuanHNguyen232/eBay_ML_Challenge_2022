import torch
import transformers

import sys
sys.path.append('../')
import utils.util as util
import config.config as cfg
import model.losses as losses


class BERTModel(torch.nn.Module):
  def __init__(self, num_labels, isFreezeBERT=True):
    super(BERTModel, self).__init__()
    self.num_labels = num_labels
    self.model_name = 'bert'

    self.bert = transformers.BertModel.from_pretrained(cfg.MODEL.BERT_ver, return_dict=True)
    self.bert_drop = torch.nn.Dropout(0.3)
    self.bert_linear = torch.nn.Linear(list(self.bert.modules())[-2].out_features, self.num_labels)  # get output of last layer (before tanh) = 768

    if isFreezeBERT:
      for param in self.bert.parameters():
        param.requires_grad = False

  def forward(self, ids, masks, token_type_ids, target_tags):
    '''
    bert ids, masks, type_ids must have shape (BATCH_SIZE, MAX_SIZE)
    '''
    out1 = self.bert(ids, attention_mask=masks, token_type_ids=token_type_ids)
    out2 = self.bert_drop(out1.last_hidden_state)
    out3 = self.bert_linear(out2)
    
    loss = losses.loss_fn(out3, target_tags, masks, self.num_labels)
    
    return out3, loss