import torch

def loss_fn(pred, target, mask, num_labels):
  lfn = torch.nn.CrossEntropyLoss()
  active_loss = mask.view(-1)==1
  
  active_logits = pred.view(-1, num_labels)
  active_labels = torch.where(
      active_loss,
      target.view(-1),
      torch.tensor(lfn.ignore_index).type_as(target)
  )
  loss = lfn(active_logits, active_labels)
  return loss