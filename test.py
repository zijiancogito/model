#!(CUDA_VISIBLE_DEVICES=-1)
from param import *
from data_iterator import MyDataset, MyIterator
from model_utils import make_model
from my_decode import greedy_decode

from torchtext import data, datasets
import torch
import pandas as pd
import numpy as np
import os

INS_SPLIT = '<nop>'
BLANK_WORD = '<blank>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

tokenize = lambda x: x.split(' ')

SRC = data.Field(sequential=True, tokenize=tokenize, pad_token=BLANK_WORD, lower=True)
TGT = data.Field(sequential=True, tokenize=tokenize, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD, lower=True)


train = MyDataset(datafile=TRAIN_FILE, asm_field=SRC, ast_field=TGT)
test = MyDataset(datafile=TEST_FILE, asm_field=SRC, ast_field=TGT)

SRC.build_vocab(train)
TGT.build_vocab(train)

src_pad_idx = SRC.vocab.stoi["<blank>"]
tgt_pad_idx = TGT.vocab.stoi["<blank>"]
split_idx = SRC.vocab.stoi['<nop>']

print("Loading model...")
model = make_model(len(SRC.vocab),
                   len(TGT.vocab),
                   src_token_len=SRC_TOKEN_LEN,
                   token=SRC_TOKEN,
                   ins_pad=split_idx,
                   pad_idx=src_pad_idx,
                   N=LAYER_NUM,
                   d_model=D_MODEL,
                   h=H)

model.load_state_dict(torch.load('model-5.pt', map_location=torch.device('cpu')))

test_iter = MyIterator(test, 
                       batch_size=BATCH_SIZE,
                       repeat=False,
                       sort_key=lambda x: x.src.count(INS_SPLIT),
                       train=False)

field = ["asm_length", "ast_length", "asm", "target", "translation"]

count=0

for i, batch in enumerate(test_iter):
  src = batch.src.transpose(0, 1)[:1]
  shape = src.shape
  tmp_src_mask = (src != src_pad_idx).unsqueeze(-2).reshape([shape[0], int(shape[1]/SRC_TOKEN), SRC_TOKEN]).sum(dim=-1)
  mask = (tmp_src_mask != 0).unsqueeze(-2)
  out = greedy_decode(model, src, mask, 
                      max_len=MAX_LEN, 
                      start_symbol=TGT.vocab.stoi["<s>"])
  print("Translation:", end="\t")
  trans = []
  for j in range(1, out.size(1)):
    # print(out[0,i])
    sym = TGT.vocab.itos[out[0, j]]
    if sym == "</s>": 
      trans.append(sym)
      print("</s>")
      break
    print(sym, end=" ")
    trans.append(sym)
  print()
  print("Target:", end="\t")
  target = []
  for j in range(1, batch.trg.size(0)):
    sym = TGT.vocab.itos[batch.trg.data[j, 0]]
    if sym == "</s>":
      target.append(sym)
      break
    print(sym, end=" ")
    target.append(sym)
  print()
  print()
  asm = []
  for index in src[0]:
    if index == src_pad_idx:
      break
    if index == split_idx:
      continue
    asm.append(SRC.vocab.itos[index])
  print(asm)
  dt = [[int(len(asm)/8), len(target), ' '.join(asm), ' '.join(target), ' '.join(trans)]]
  data = pd.DataFrame(columns=field, data=dt)
  if not os.path.exists('translation.csv'):
    data.to_csv('translation.csv', mode='a+', encoding='utf-8', header=True)
  else:
    data.to_csv('translation.csv', mode='a', encoding='utf-8', header=False)
  # break
  count+=1

  from visualization import draw, visualization
  visualization(model, trans, ' '.join(asm))
  break