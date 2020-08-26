import collections
import math
import numpy as np
import torch

def _get_ngrams(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams up to max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts

def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matched_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)
    overlap = dict((ngram, min(count, translation_ngram_counts[ngram]))
                    for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matched_by_order[len(ngram)-1] += translation_ngram_counts[ngram]
    precisions = [0] * max_order
    smooth = 1.0
    for i in range(0, max_order):
      if possible_matched_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matched_by_order[i]
        if matches_by_order[i] > 0:
          precisions[i] = matches_by_order[i] / possible_matched_by_order[i]
        else:
          smooth *= 2
          precisions[i] = 1.0 / (smooth * possible_matched_by_order[i])
      else:
        precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum/max_order)

  if use_bp:
    if not reference_length:
      bp = 1.0
    else:
      ratio = translation_length / reference_length
      if ratio <= 0.0:
        bp = 0.0
      elif ratio >= 1.0:
        bp = 1.0
      else:
        bp = math.exp(1 - 1. / ratio)
  bleu = geo_mean * bp
  return np.float32(bleu)

def bleu_score(predictions, labels, **unused_kwargs):
  outputs = predictions
  import pdb
  pdb.set_trace()
  # outputs = torch.squeeze(outputs, dim=[-1, -2])
  # labels = torch.squeeze(labels, dim=[-1, -2])

  bleu = compute_bleu(labels, outputs)
  
  print(bleu)
  return bleu, 1.0
