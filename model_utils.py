import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, time, math
from torch.autograd import Variable

from layers import PositionwiseFeedForward
from layers import Encoder, EncoderLayer
from layers import Decoder, DecoderLayer
from layers import Generator
from embedding import Embeddings
from model_arch import EncoderDecoder
from attention import MultiHeadedAttention
from position import PositionalEncoding

def make_model(src_vocab, 
               tgt_vocab, 
               src_token_len,
               token,
               ins_pad,
               pad_idx,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1):
  c = copy.deepcopy
  attn = MultiHeadedAttention(h, d_model)
  ff = PositionwiseFeedForward(d_model, d_ff, dropout)
  position = PositionalEncoding(d_model, dropout)
  model = EncoderDecoder(
    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
    Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
    nn.Sequential(Embeddings(d_model, src_vocab, src_token_len, token, ins_pad, pad_idx), c(position)),
    nn.Sequential(Embeddings(d_model, tgt_vocab, pad_idx=pad_idx), c(position)),
    Generator(d_model, tgt_vocab)
  )
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform(p)
  return model

def run_epoch(data_iter, model, loss_compute):
  start = time.time()
  total_tokens = 0
  total_loss = 0
  tokens = 0
  total_correct = 0.0
  total_valid = 1.0
  for i, batch in enumerate(data_iter):
    out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
    loss, ncorrect, nvalid = loss_compute(out, batch.trg_y, batch.ntokens)
    total_loss += loss
    total_tokens += batch.ntokens
    total_correct += ncorrect
    total_valid += nvalid
    tokens += batch.ntokens
    if i % 50 == 1:
      elapsed = time.time() - start
      print("Epoch Step: %d Loss: %f Tokens per Sec: %f Accu: %f" %
              (i, loss / batch.ntokens, tokens / elapsed, total_correct/total_valid*100))
      start = time.time()
      tokens = 0
  return total_loss / total_tokens, total_correct / total_valid * 100
