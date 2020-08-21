import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

def average_vector(vectors):
  from numpy import mean
  if len(vectors) > 1:
    ret = vectors[0] + mean(vectors[1:], axis=0).tolist()
  else:
    ret = vectors[0] + vectors[0]
  return ret

def average_emb(emb, x, split_index, pad_idx):
  tmp_emb = []
  tmp_vector = []
  for vector, index in zip(emb,x):
    if index == split_index:
      tmp_emb.append(average_vector(tmp_vector))
      tmp_vector = []
    if index == pad_idx:
      break
    else:
      tmp_vector.append(vector)
  return tmp_emb

def average_batch(p_batch, p_x, split_index, pad_idx, p_pad_emb):
  pad_emb = p_pad_emb.cpu()
  batch = p_batch.cpu()
  x = p_x.cpu()
  pad_emb = pad_emb.detach().numpy().tolist()
  batch = batch.detach().numpy().tolist()
  x = x.detach().numpy().tolist()
  t_batch_list = []
  for emb, index in zip(batch, x):
    t_batch_list.append(average_emb(emb, index, split_index, pad_idx))
  # max_len = max([len(emb) for emb in t_batch_list])
  length = max([len(emb) for emb in t_batch_list])
  # if max_len == min_len:
  #   max_len += 1
  max_len = p_batch.shape[-2] // 2
  batch_list = []
  for emb in t_batch_list:
    if len(emb) < max_len:
      tmp_emb = emb + (max_len - len(emb)) * pad_emb
      batch_list.append(tmp_emb)
    else:
      batch_list.append(emb)
  new_batch = torch.Tensor(batch_list)
  if torch.cuda.is_available():
    new_batch_cuda = new_batch.cuda()
  else:
    new_batch_cuda = new_batch
  return new_batch_cuda

def avg_batch(emb, src, ins_pad, token):
  mask = (src != ins_pad).unsqueeze(-1)
  tmp_emb = emb * mask
  shape = emb.shape
  tmp_emb = tmp_emb.reshape(shape[0], int(emb.shape[1]/token), token, shape[2])
  opc = torch.index_select(tmp_emb, -2, torch.LongTensor([0]).cuda())
  ops = torch.index_select(tmp_emb, -2, torch.LongTensor(list(range(1,token))).cuda())
  ops_avg = torch.mean(ops, dim=-2, keepdim=True)
  concat_op = torch.cat((opc, ops_avg), -1)
  return concat_op.squeeze(-2)

class Embeddings(nn.Module):
  def __init__(self, d_model, vocab, token_len=1, token=0, ins_pad=0, pad_idx=0):
    super(Embeddings, self).__init__()
    self.lut = nn.Embedding(vocab, int(d_model/token_len))
    self.d_model = d_model
    self.token = token
    self.token_len = token_len
    self.ins_pad = ins_pad
    self.vocab = vocab
    self.pad_idx = pad_idx

  def forward(self, x):
    tmp_emb = self.lut(x) * math.sqrt(int(self.d_model / self.token_len))
    if self.token_len > 1:
      emb = avg_batch(tmp_emb, x, self.ins_pad, self.token)
    else:
      emb = tmp_emb
    return emb