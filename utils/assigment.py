from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
import torch.nn.functional as F

def cdist_vary_lengths(A, A_n, B, B_n, p=2):
    """
    Args:
        A: (n, dim)
        A_n: (bs)
        B: (m, dim)
        B_n: (bs)
    """
  # (bs, t1, dim)
  A = chunk_pad_by_lengths(A, A_n, batch_first=True).type(torch.float32)
  # (bs, t2, dim)
  B = chunk_pad_by_lengths(B, B_n, batch_first=True).type(torch.float32)
  # (bs, t1, t2)
  dists = torch.cdist(A, B, p=p)
  return dists