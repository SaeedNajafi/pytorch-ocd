"""
Implementation of useful modules.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np

VERY_LARGE_NUM = 1e+8

def edit_distance_q_values(sampled_y, gold_y, gold_y_mask, vocab_size):
    """ OCD Edit Distance to compute QValues. 
    Args:
        sampled_y (LongTensor): ``(batch_size, sequence_length)``
        gold_y (LongTensor): ``(batch_size, sequence_length)``
        gold_y_mask: ``(batch_size, sequence_length)``
        vocab_size (integer): number of possible ouput tokens at each step.
    Returns:
        FloatTensor:
        * q_values ``(batch_size, sequence_length, vocab_size)``
    """
    b_sz, seq_len = gold_y.size()
    q_values = gold_y.new_zeros(b_sz, seq_len, vocab_size)
    edit_dists = gold_y.new_zeros(b_sz, seq_len + 1, seq_len + 1)
    edit_dists[:, :, 0] = torch.arange(seq_len + 1)
    edit_dists[:, 0, :] = torch.arange(seq_len + 1)
    for i in range(1, seq_len + 1):
        for j in range(1, seq_len + 1):
            cost = sampled_y[:, i-1] == gold_y[:, j-1]
            edit_dists[:, i, j] = torch.cat((edit_dists[:, i-1, j] + 1, edit_dists[:, i, j-1] + 1,
                                             edit_dists[:, i-1, j-1] + cost), dim=1).min(dim=1)
    edit_dists = edit_dists.masked_fill_(gold_y_mask.unsqueeze(dim=1), VERY_LARGE_NUM)
    min_dist = edit_dists.min(dim=2).unsqueeze(dim=2)
    steps_with_min_dists = edit_dists == min_dist
    extended_gold_y = gold_y.repeat(1, 2).view(b_sz, seq_len, seq_len)
    indices = steps_with_min_dists.nonzero().split(1, dim=1)
    gold_next_tokens = extended_gold_y[indices]
    indices[:, 2] = gold_next_tokens
    q_values[indices] = 1
    q_values = q_values - (1 + min_dist)
    return q_values

def compute_optimal_pi(q_values: torch.FloatTensor, 
                       theta: Optional[float] = 1.0/VERY_LARGE_NUM) -> torch.FloatTensor:
    assert theta >= 0
    if theta == 0:
        theta = 1.0/VERY_LARGE_NUM

    normalized_q_values = normalized_q_values / theta
    return nn.functional.softmax(normalized_q_values, dim=2)

def ocd_loss(optimal_pi: torch:FloatTensor,
             model_log_probs: torch.FloatTensor,
             sampled_y_mask: torch.ByteTensor):

    loss = nn.functional.kl_div(model_log_probs, optimal_pi, reduction='none')
    loss *= 1 - sampled_y_mask
    loss = loss.sum(dim=1)
    return loss.mean(dim=0)
