import torch
import torch.nn as nn
from torch.autograd import Variable

class SimpleLossCompute:
  def __init__(self, generator, criterion, accuracy, opt=None, devices=[-1]):
    self.generator = generator
    self.criterion = criterion
    self.accuracy = accuracy
    self.opt = opt
  
  def __call__(self, x, y, norm):
    x = self.generator(x)
    loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                          y.contiguous().view(-1)) / norm
    _, x_pred = x.contiguous().view(-1, x.size(-1)).max(dim=-1)
    y_true = y.contiguous().view(-1)
    n_correct, n_valid = self.accuracy(x_pred, y_true)
    loss.backward()
    if self.opt is not None:
      self.opt.step()
      self.opt.optimizer.zero_grad()
    return loss.data * norm, n_correct, n_valid

class MultiGPULossCompute:
  def __init__(self, generator, criterion, accuracy, devices, opt=None, chunk_size=4):
    self.generator = generator
    self.criterion = nn.parallel.replicate(criterion, 
                                           devices=devices)
    self.opt = opt
    self.devices = devices
    self.chunk_size = chunk_size
    self.accuracy = nn.parallel.replicate(accuracy, devices=devices)

  def __call__(self, out, targets, normalize):
    total = 0.0
    generator = nn.parallel.replicate(self.generator, devices=self.devices)
    out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
    out_grad = [[] for _ in out_scatter]
    targets = nn.parallel.scatter(targets, target_gpus=self.devices)

    chunk_size = self.chunk_size
    total_correct = 0
    total_valid = 0
    for i in range(0, out_scatter[0].size(1), chunk_size):
      out_column = [[Variable(o[:, i:i+chunk_size].data,
                              requires_grad=self.opt is not None)]
                     for o in out_scatter]
      gen = nn.parallel.parallel_apply(generator, out_column)
      y = [(g.contiguous().view(-1, g.size(-1)),
            t[:, i:i+chunk_size].contiguous().view(-1))
          for g, t in zip(gen, targets)]
      
      loss = nn.parallel.parallel_apply(self.criterion, y)

      result = nn.parallel.parallel_apply(self.accuracy, y)
      # import pdb
      # pdb.set_trace()
      n_correct = [i[0] for i in result]
      n_valid = [i[1] for i in result]
      nc = nn.parallel.gather(n_correct, target_device=self.devices[0])
      nv = nn.parallel.gather(n_valid, target_device=self.devices[0])

      l = nn.parallel.gather(loss, target_device=self.devices[0])
      l = l.sum() / normalize
      total += l.data

      total_correct += nc.data
      total_valid += nv.data
      if self.opt is not None:
        l.backward()
        for j, l in enumerate(loss):
          out_grad[j].append(out_column[j][0].grad.data.clone())
    
    if self.opt is not None:
      out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
      o1 = out
      o2 = nn.parallel.gather(out_grad,
                              target_device=self.devices[0])
      o1.backward(gradient=o2)
      self.opt.step()
      self.opt.optimizer.zero_grad()
    return total * normalize, total_correct, total_valid
