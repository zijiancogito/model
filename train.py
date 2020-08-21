from torchtext import data, datasets
from param import *
from data_iterator import MyDataset, MyIterator
from model_utils import make_model, run_epoch
from label import LabelSmoothing
from data_iterator import batch_size_fn, rebatch
from opt import NoamOpt, get_std_opt
from loss import MultiGPULossCompute

import torch.nn as nn
import torch

tokenize = lambda x: x.split(' ')

BLANK_WORD = '<blank>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
SPLIT_WORD = '<split>'

SRC = data.Field(sequential=True, tokenize=tokenize, pad_token=BLANK_WORD, lower=True)

TGT = data.Field(sequential=True, tokenize=tokenize, init_token=BOS_WORD,eos_token=EOS_WORD, pad_token=BLANK_WORD, lower=True)

train = MyDataset(datafile=TRAIN_FILE, asm_field=SRC, ast_field=TGT)
val = MyDataset(datafile=VAL_FILE, asm_field=SRC, ast_field=TGT)

SRC.build_vocab(train)
TGT.build_vocab(train)

src_pad_idx = SRC.vocab.stoi[BLANK_WORD]
tgt_pad_idx = TGT.vocab.stoi[BLANK_WORD]
split_idx = SRC.vocab.stoi[SPLIT_WORD]

model = make_model(len(SRC.vocab),
                   len(TGT.vocab),
                   src_token_len=SRC_TOKEN_LEN,
                   trg_token_len=TRG_TOEKN_LEN,
                   split_idx=split_idx,
                   pad_idx=src_pad_idx,
                   N=LAYER_NUM,
                   d_model=D_MODEL,
                   h=H)
model.cuda()
criterion = LabelSmoothing(size=len(TGT.vocab),
                           padding_idx=tgt_pad_idx,
                           smoothing=LABEL_SMOOTH)
criterion.cuda()
train_iter = MyIterator(train,
                        batch_size=BATCH_SIZE,
                        device=0,
                        repeat=True,
                        sort=True,
                        sort_key=lambda x: x.src.count(SPLIT_WORD),
                        batch_size_fn=batch_size_fn,
                        train=True)
valid_iter = MyIterator(val,
                        batch_size=BATCH_SIZE,
                        device=0,
                        repeat=True,
                        sort=True,
                        sort_key=lambda x: x.src.count(SPLIT_WORD),
                        batch_size_fn=batch_size_fn,
                        train=False)
model_par = nn.DataParallel(model, device_ids=devices)
model_opt = get_std_opt(model)

for epoch in range(10):
  model_par.train()
  print("Train:")
  run_epoch((rebatch(src_pad_idx,
                     tgt_pad_idx,
                     split_idx,
                     b,
                     src_token_len=SRC_TOKEN_LEN,
                     trg_token_len=TRG_TOEKN_LEN,
                    )
                     for b in train_iter),
             model_par,
             MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt)
  )
  model_par.eval()
  print("Eval:")
  loss = run_epoch((rebatch(src_pad_idx,
                            tgt_pad_idx,
                            split_idx,
                            b,
                            src_token_len=SRC_TOKEN_LEN,
                            trg_token_len=TRG_TOEKN_LEN,
                            )
                            for b in valid_iter),
                    model_par,
                    MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt)
  )
  torch.save(model.state_dict(), f'model-{epoch}.pt')