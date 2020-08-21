from param import *
from data_iterator import MyDataset, MyIterator
from model_utils import make_model
from data_iterator import src_mask
from my_decode import greedy_decode

from torchtext import data, datasets
import torch

INS_SPLIT = '<split>'
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

devices = [0, 1, 2, 3, 4]
src_pad_idx = SRC.vocab.stoi["<blank>"]
tgt_pad_idx = TGT.vocab.stoi["<blank>"]
split_idx = SRC.vocab.stoi['<split>']

print("Loading model...")
model = make_model(len(SRC.vocab), 
                   len(TGT.vocab), 
                   src_token_len=SRC_TOKEN_LEN, trg_token_len=TRG_TOEKN_LEN,
                   split_idx=split_idx,
                   pad_idx=src_pad_idx,
                   N=LAYER_NUM,
                   d_model=D_MODEL, 
                   h=H)

model.load_state_dict(torch.load('model/model-4.pt', map_location=torch.device('cpu')))

test_iter = MyIterator(test, 
                       batch_size=BATCH_SIZE,
                       repeat=False,
                       sort_key=lambda x: x.src.count(INS_SPLIT),
                       train=False)

field = ["asm_length", "ast_length", "asm", "target", "translation"]

count=0

for i, batch in enumerate(test_iter):
  for j in batch.src.transpose(0, 1):
    src = tmp_src = src_mask(src, split_idx, src_pad_idx)
    mask = (tmp_src != src_pad_idx).unsqueeze(-2)
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
      asm.append(SRC.vocab.itos[index])
    dt = [[int(len(asm)/8), len(target), ' '.join(asm), ' '.join(target), ' '.join(trans)]]
    data = pd.DataFrame(columns=field, data=dt)
    if not os.path.exists('translation.csv'):
      data.to_csv('translation.csv', mode='a+', encoding='utf-8', header=True)
    else:
      data.to_csv('translation.csv', mode='a', encoding='utf-8', header=False)
    # break
    count+=1