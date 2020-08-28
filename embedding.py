import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

def avg_batch(emb, src, ins_pad, token):
  mask = (src != ins_pad).unsqueeze(-1)
  tmp_emb = emb * mask
  shape = emb.shape
  tmp_emb = tmp_emb.reshape(shape[0], int(emb.shape[1]/token), token, shape[2])
  index1 = torch.LongTensor([0]).cuda() if torch.cuda.is_available() else torch.LongTensor([0])
  opc = torch.index_select(tmp_emb, -2, index1)
  index2 = torch.LongTensor(list(range(1,token))).cuda() if torch.cuda.is_available() else torch.LongTensor(list(range(1,token)))
  ops = torch.index_select(tmp_emb, -2, index2)
  ops_avg = torch.mean(ops, dim=-2, keepdim=True)
  concat_op = torch.cat((opc, ops_avg), -1)
  del ops_avg, ops, index1, index2, opc
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
    emb = self.lut(x) * math.sqrt(int(self.d_model / self.token_len))
    if self.token_len > 1:
      emb = avg_batch(emb, x, self.ins_pad, self.token)
    return emb