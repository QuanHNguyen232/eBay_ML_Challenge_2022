import torch
import transformers

import sys
sys.path.append('../')
import utils.util as util
import config.config as cfg
import model.losses as losses

# def loss_fn(pred, target, mask, num_labels):
#   lfn = torch.nn.CrossEntropyLoss()
#   active_loss = mask.view(-1)==1
  
#   active_logits = pred.view(-1, num_labels)
#   active_labels = torch.where(
#       active_loss,
#       target.view(-1),
#       torch.tensor(lfn.ignore_index).type_as(target)
#   )
#   loss = lfn(active_logits, active_labels)
#   return loss

class MyModel(torch.nn.Module):
  def __init__(self, num_labels):
    super(MyModel, self).__init__()
    self.num_labels = num_labels

    self.bert = transformers.BertModel.from_pretrained(cfg.MODEL.BERT_ver, return_dict=True)
    self.bert_drop = torch.nn.Dropout(0.3)
    self.bert_linear = torch.nn.Linear(list(self.bert.modules())[-2].out_features, self.num_labels)  # get output of last layer (before tanh) = 768

  def forward(self, ids, masks, token_type_ids, target_tags):
    out1 = self.bert(ids, attention_mask=masks, token_type_ids=token_type_ids)
    out2 = self.bert_drop(out1.last_hidden_state)
    out3 = self.bert_linear(out2)
    
    loss = losses.loss_fn(out3, target_tags, masks, self.num_labels)
    
    return out3, loss