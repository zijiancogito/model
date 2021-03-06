import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

def draw(data, x, y, ax):
  seaborn.heatmap(data,
                  xticklabels=x,
                  square=True,
                  yticklabels=y,
                  vmin=0.0,
                  vmax=1.0,
                  cbar=False,
                  ax=ax)

def visualization(model, trans, sent):
  tgt_sent = trans

  for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1, 4, figsize=(1024, 512))
    print("Encoder Layer", layer+1)
    for h in range(4):
      draw(model.encoder.layers[layer].self_attn.attn[0, h].data, sent, sent if h == 0 else [], ax=axs[h])
    # plt.show()
    plt.savefig(f'./encode_self_layer_{layer}.jpg')

  for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1, 4, figsize=(1024, 512))
    print("Decoder Self Layer", layer+1)
    for h in range(4):
      draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], tgt_sent, tgt_sent if h == 0 else [], ax=axs[h])
    # plt.show()
    plt.savefig(f'./decode_self_layer_{layer}.jpg')
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1, 4, figsize=(1024, 512))
    for h in range(4):
      draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], sent, tgt_sent if h ==0 else [], ax=axs[h])
    # plt.show()
    plt.savefig(f'./decode_src_layer_{layer}.jpg')

