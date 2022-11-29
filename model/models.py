import torch
import transformers

import sys
sys.path.append('../')
import utils.util as util
import config.config as cfg
import model.losses as losses
from model.nn import ConditionalRandomField

class BERTModel(torch.nn.Module):
  def __init__(self, num_labels, isFreeze=True):
    super(BERTModel, self).__init__()
    self.num_labels = num_labels
    self.model_name = 'roberta'
    self.isFreeze = isFreeze

    self.bert = transformers.BertModel.from_pretrained(cfg.MODEL.RoBERTa_ver)
    # self.bert = transformers.BertModel.from_pretrained(cfg.MODEL.BERT_ver, return_dict=True)

    self.drop1 = torch.nn.Dropout(0.3)
    self.linear1 = torch.nn.Linear(list(self.bert.modules())[-2].out_features, self.num_labels)  # get output of last layer (before tanh) = 768
    self.relu1 = torch.nn.ReLU()
    self.drop2 = torch.nn.Dropout(0.5)
    self.linear2 = torch.nn.Linear(self.num_labels, self.num_labels)

    if self.isFreeze:
      for param in self.bert.parameters():
        param.requires_grad = False

  def forward(self, ids, masks, token_type_ids, target_tags=None):
    '''
    bert ids, masks, type_ids must have shape (BATCH_SIZE, MAX_SIZE)
    '''
    out = self.bert(ids, attention_mask=masks, token_type_ids=token_type_ids)
    out = self.drop1(out.last_hidden_state)
    out = self.linear1(out)
    out = self.relu1(out)
    out = self.drop2(out)
    out = self.linear2(out)
    
    if target_tags != None:
      loss = losses.loss_fn(out, target_tags, masks, self.num_labels)
      return out, loss
    
    return out

class BERTModel_1(torch.nn.Module):
  def __init__(self, num_labels, isFreeze=True):
    super(BERTModel_1, self).__init__()
    self.num_labels = num_labels
    self.model_name = 'bert'
    self.isFreeze = isFreeze

    self.bert = transformers.BertModel.from_pretrained(cfg.MODEL.BERT_ver)
    self.bert_drop = torch.nn.Dropout(0.3)
    self.bert_linear = torch.nn.Linear(list(self.bert.modules())[-2].out_features, self.num_labels)  # get output of last layer (before tanh) = 768

    if self.isFreeze:
      for param in self.bert.parameters():
        param.requires_grad = False

  def forward(self, ids, masks, token_type_ids, target_tags=None):
    '''
    bert ids, masks, type_ids must have shape (BATCH_SIZE, MAX_SIZE)
    '''
    out = self.bert(ids, attention_mask=masks, token_type_ids=token_type_ids)
    out = self.bert_drop(out.last_hidden_state)
    out = self.bert_linear(out)
    
    if target_tags != None:
      loss = losses.loss_fn(out, target_tags, masks, self.num_labels)
      return out, loss
    
    return out

class RoBERTa_BiLSTM_Model(torch.nn.Module):
  def __init__(self, num_labels, isFreeze=False, lstm_hidden_dim=256):
    super(RoBERTa_BiLSTM_Model, self).__init__()
    self.num_labels = num_labels
    self.model_name = 'roberta-bilstm'
    self.isFreeze = isFreeze
    self.bert_cfg = transformers.BertConfig.from_pretrained(cfg.MODEL.RoBERTa_ver)

    self.roberta = transformers.BertModel.from_pretrained(cfg.MODEL.RoBERTa_ver)
    self.drop1 = torch.nn.Dropout(0.5)
    self.drop2 = torch.nn.Dropout(0.5)
    self.bilstm = torch.nn.LSTM(self.bert_cfg.hidden_size, lstm_hidden_dim // 2, num_layers=2, bidirectional=True, dropout=0.1, batch_first=True)
    self.linear1 = torch.nn.Linear(lstm_hidden_dim, self.num_labels)  # get output of last layer (before tanh) = 768

    if self.isFreeze:
      for param in self.roberta.parameters():
        param.requires_grad = False

  def forward(self, ids, masks, token_type_ids, target_tags=None):
    out = self.roberta(ids, attention_mask=masks, token_type_ids=token_type_ids)
    out = self.drop1(out.last_hidden_state)
    out, _ = self.bilstm(out)
    out = self.drop2(out)
    out = self.linear1(out)
    
    if target_tags != None:
      loss = losses.loss_fn(out, target_tags, masks, self.num_labels)
      return out, loss

    return out




class RoBERTa_BiLSTM_CRF_Model(torch.nn.Module):  # FAILED
  def __init__(self, num_labels, isFreeze=True, lstm_hidden_dim=256):
    super(RoBERTa_BiLSTM_CRF_Model, self).__init__()
    self.num_labels = num_labels
    self.model_name = 'roberta-bilstm-crf'
    self.isFreeze = isFreeze
    self.bert_cfg = transformers.BertConfig.from_pretrained(cfg.MODEL.RoBERTa_ver)

    self.roberta = transformers.BertModel.from_pretrained(cfg.MODEL.RoBERTa_ver)
    self.drop1 = torch.nn.Dropout(0.3)
    self.bilstm = torch.nn.LSTM(self.bert_cfg.hidden_size, lstm_hidden_dim // 2, num_layers=2, bidirectional=True, dropout=0.1, batch_first=True)
    self.linear1 = torch.nn.Linear(lstm_hidden_dim, self.num_labels)  # get output of last layer (before tanh) = 768
    self.crf = ConditionalRandomField(self.num_labels)

    if self.isFreeze:
      for param in self.roberta.parameters():
        param.requires_grad = False

  def forward(self, ids, masks, token_type_ids, target_tags):
    out = self.roberta(ids, attention_mask=masks, token_type_ids=token_type_ids)
    out, _ = self.bilstm(out.last_hidden_state)
    out = self.linear1(out)
        
    # For CRF, dataset: special_tk_tag = list(cfg.encode_tag.classes_).index('No Tag')
    loss = self.crf.neg_log_likelihood_loss(out, masks.to(torch.bool), target_tags)

    return self.crf(out, masks.to(torch.bool)), loss

