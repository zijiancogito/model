import torch
import torch.nn as nn

class ComputeAccuracy(nn.Module):
  def __init__(self, mask_index=0):
    super(ComputeAccuracy, self).__init__()
    self.mask_index = mask_index

  def forward(self, y_pred, y_true):
    _, y_pred_indices = y_pred.max(dim=1)
    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, self.mask_index).float()
    n_correct = (correct_indices * valid_indices).sum().squeeze()
    n_valid = valid_indices.sum().squeeze()
    return torch.stack((n_correct, n_valid)).unsqueeze(dim=0)

def accuracy(y_pred, y_true, mask_index):
  # _, y_pred_indices = y_pred.max(dim=1)
  correct_indices = torch.eq(y_pred, y_true).float()
  valid_indices = torch.ne(y_true, mask_index).float()
  n_correct = (correct_indices * valid_indices).sum()
  n_valid = valid_indices.sum()
  return n_correct, n_valid
