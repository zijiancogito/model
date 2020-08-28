from torchtext import data, datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import pandas as pd

def subsequent_mask(size):
  attn_shape = (1, size, size)
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  return torch.from_numpy(subsequent_mask) == 0

class Batch:
  def __init__(self, src, token, ins_pad, trg=None, src_pad=0, tgt_pad=0):
    self.src = src
    shape = self.src.shape
    tmp_src_mask = (src != src_pad).unsqueeze(-2).reshape([shape[0], int(shape[1]/token), token]).sum(dim=-1)
    self.src_mask = (tmp_src_mask != 0).unsqueeze(-2)
    del tmp_src_mask
    if trg is not None:
      self.trg = trg[:, :-1]
      self.trg_y = trg[:, 1:]
      self.trg_mask = \
          self.make_std_mask(self.trg, tgt_pad)
      self.ntokens = (self.trg_y != tgt_pad).data.sum()

  @staticmethod
  def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
  global max_src_in_batch, max_tgt_in_batch
  if count == 1:
    max_src_in_batch = 0
    max_tgt_in_batch = 0
  max_src_in_batch = max(max_src_in_batch, len(new.src))
  max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
  src_elements = count * max_src_in_batch
  tgt_elements = count * max_tgt_in_batch
  return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
  def create_batches(self):
    if self.train:
      def pool(d, random_shuffler):
        for p in data.batch(d, self.batch_size * 100):
          p_batch = data.batch(
              sorted(p, key=self.sort_key),
              self.batch_size)
          for b in random_shuffler(list(p_batch)):
            yield b
      self.batches = pool(self.data(), self.random_shuffler)
    else:
      self.batches = []
      for b in data.batch(self.data(), self.batch_size):
        self.batches.append(sorted(b, key=self.sort_key))

def rebatch(src_pad_idx, tgt_pad_idx, ins_pad, batch, token):
  src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
  return Batch(src, token, ins_pad, trg, src_pad_idx, tgt_pad_idx)

class MyDataset(data.Dataset):
  def __init__(self, datafile, asm_field, ast_field, test=False, **kwargs):
    fields = [("src", asm_field), ("trg", ast_field)]
    examples = []
    tmp_data = pd.read_csv(datafile)
    shape = tmp_data.shape[0]
    drop = shape % 1000
    csv_data = tmp_data.drop(range(shape - drop, shape))
    del tmp_data
    if test:
      pass
    else:
      from tqdm import tqdm
      for asm, ast in tqdm(zip(csv_data['asm'], csv_data['ast'])):
        examples.append(data.Example.fromlist([asm, ast], fields))
    super(MyDataset, self).__init__(examples, fields, **kwargs)