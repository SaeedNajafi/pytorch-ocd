"""
Implementation of ocd modules.
"""
# pylint: disable=E1101
# pylint: disable=arguments-differ
# pylint: disable=too-many-locals
from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

LARGE_NUM = float(1e+8)


class OCD(nn.Module):
    """ Optimal Completion Distillation

    This class implements an Optimal Character Distillation
    precedure with negative character edit distance as reward function.
    :cite:`https://arxiv.org/abs/1810.01398`

    Args:
        end_symbol_id: index of the end symbol in your vocabulary.
        vocab_size: the number of possible output tokens.
        tau: tau parameter used to normalize QValues
             for policy calculation with softmax function.
             if not given: tau limits 0

    Attributes:
        end_symbol_id (int)
        vocab_size (int)
        tau (int)
    """
    def __init__(self, vocab_size: int,
                 end_symbol_id: int,
                 tau: Optional[float] = None) -> None:
        if vocab_size <= 0:
            raise ValueError(f'invalid vocabulary size > 0: {vocab_size}')
        if end_symbol_id < 0:
            raise ValueError(f'invalid end symbol index >=0: {end_symbol_id}')
        if tau and tau < 0:
            raise ValueError(f'invalid tau parameter >=0: {tau}')

        super().__init__()
        self.vocab_size = vocab_size
        self.end_symbol_id = end_symbol_id
        self.tau = tau

    def forward(self, model_scores: torch.FloatTensor,
                gold_y: torch.LongTensor) -> torch.Tensor:
        """ samples from the model_scores and returns the OCD loss.
        Args:
            model_scores (`~torch.FloatTensor`): ``(batch_size, sequence_lenght, vocab_size)``
                                                 scores given by the model, scores before the log softmax.
            gold_y (`~torch.LongTensor`): ``(batch_size, sequence_length)``
        Returns:
            `~torch.Tensor`: loss to backpropagate
        """

        b_sz, seq_len, vocab_size = model_scores.size()
        assert vocab_size == self.vocab_size
        log_probs = nn.functional.log_softmax(model_scores, dim=2)
        dist = Categorical(logits=log_probs.view(-1, vocab_size))
        # take one sample from the model.
        sampled_y = dist.sample().view(b_sz, seq_len)
        sampled_y_mask = OCD.sequence_mask(sampled_y, self.end_symbol_id)
        q_values = OCD.edit_distance_q_values(sampled_y, gold_y,
                                              self.end_symbol_id, self.vocab_size)
        policy = OCD.compute_optimal_pi(q_values, self.tau)
        return OCD.loss(policy, log_probs, sampled_y_mask)

    @staticmethod
    def sequence_mask(sequence: torch.LongTensor,
                      end_symbol_id: int) -> torch.ByteTensor:
        """ creates a mask tensor for the input 'sequence'.
        'end_symbol_id' indicates the end of sequence.
        Args:
            sequence (`~torch.LongTensor`): ``(batch_size, sequence_length)``
            end_symbol_id (int)
        Returns:
            `~torch.ByteTensor`: zero for actual steps,
                                 ones for masked padded irrelevant steps.

        Example:
            sequence = [[1, 2, 3, 4]
                        [5, 4, 6, 4],
                        [3, 2, 4, 0]]
            end_symbol_id = 4

            expected mask = [[0, 0, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, 0, 1]]
        """

        b_sz, seq_len = sequence.size()
        end_symbols = (sequence == end_symbol_id)
        pattern = end_symbols.cumsum(dim=1) >= 1
        lens = (1 - pattern).sum(dim=1)
        final_lens = lens + 1
        mask = torch.arange(seq_len).expand(b_sz, seq_len) >= final_lens.unsqueeze(1)
        return mask

    @staticmethod
    def edit_distance_mask(sequence: torch.LongTensor,
                           end_symbol_id: int) -> torch.ByteTensor:
        """ creates a mask 3D tensor for the input 'sequence'.
        'end_symbol_id' indicates the end of sequence.
        Args:
            sequence (`~torch.LongTensor`): ``(batch_size, sequence_length)``
            end_symbol_id (int)
        Returns:
            `~torch.ByteTensor`: zero for actual steps,
                                 ones for masked padded irrelevant steps.

        Example:
            sequence = [[1, 2, 3, 4]
                        [5, 4, 6, 4],
                        [3, 2, 4, 0]]
            end_symbol_id = 4

            expected edit_mask = [[[0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 1]],

                                  [[0, 0, 1, 1, 1],
                                   [0, 0, 1, 1, 1],
                                   [0, 0, 1, 1, 1],
                                   [0, 0, 1, 1, 1],
                                   [0, 0, 1, 1, 1]],

                                  [[0, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 1],
                                   [0, 0, 0, 1, 1]]]
        """

        b_sz, seq_len = sequence.size()
        mask = OCD.sequence_mask(sequence, end_symbol_id)

        # post-append mask for a sequence of length 'seq_len + 1'
        added_ones = mask.new_zeros(b_sz, dtype=torch.uint8) + 1
        mask = torch.cat((mask, added_ones.unsqueeze(dim=1)), dim=1)
        edit_mask = mask.repeat(1, seq_len + 1).view(b_sz, seq_len + 1, seq_len + 1)
        return edit_mask

    @staticmethod
    def edit_distance_q_values(sampled_y: torch.LongTensor,
                               gold_y: torch.LongTensor,
                               end_symbol_id: int,
                               vocab_size: int) -> torch.FloatTensor:
        """ OCD Edit Distance to compute QValues.
        Args:
            sampled_y (`~torch.LongTensor`): ``(batch_size, sequence_length)``
            gold_y (`~torch.LongTensor`): ``(batch_size, sequence_length)``
            end_symbol_id (int): index of the end symbol in your vocabulary.
            vocab_size (int): the number of possible output tokens.

        Returns:
            `~torch.FloatTensor`: ``(batch_size, sequence_length, vocab_size)``

        Example:
            from paper `https://arxiv.org/abs/1810.01398`

            vocabulary = {'S':0, 'U':1, 'N':2, 'D':3, 'A':4 , 'Y':5,
                          'T':6, 'R':7, 'P':8, '</s>':9, '<pad>': 10}
            vocab_size = 11
            end_symbol_id = 9

            gold Y = {'SUNDAY</s><pad><pad>', 'SUNDAY</s><pad><pad>'}
            gold_y = [[0, 1, 2, 3, 4, 5, 9, 10, 10],
                      [0, 1, 2, 3, 4, 5, 9, 10, 10]]

            sampled Y = {'SATURDAY</s>', 'SATRAPY</s>U'}
            sampled_y = [[0, 4, 6, 1, 7, 3, 4, 5, 9],
                         [0, 4, 6, 7, 4, 8, 5, 9, 1]]


            # expected size: (batch_size=2, sequence_lenght=9, vocab_size=11)
            expected q_values = [[[ 0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
                                  [-1.,  0., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
                                  [-2., -1., -1., -2., -2., -2., -2., -2., -2., -2., -2.],
                                  [-3., -2., -2., -2., -3., -3., -3., -3., -3., -3., -3.],
                                  [-3., -3., -2., -3., -3., -3., -3., -3., -3., -3., -3.],
                                  [-4., -4., -3., -3., -4., -4., -4., -4., -4., -4., -4.],
                                  [-4., -4., -4., -4., -3., -4., -4., -4., -4., -4., -4.],
                                  [-4., -4., -4., -4., -4., -3., -4., -4., -4., -4., -4.],
                                  [-4., -4., -4., -4., -4., -4., -4., -4., -4., -3., -4.]],

                                 [[ 0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
                                  [-1.,  0., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
                                  [-2., -1., -1., -2., -2., -2., -2., -2., -2., -2., -2.],
                                  [-3., -2., -2., -2., -3., -3., -3., -3., -3., -3., -3.],
                                  [-4., -3., -3., -3., -3., -4., -4., -4., -4., -4., -4.],
                                  [-4., -4., -4., -4., -4., -3., -4., -4., -4., -4., -4.],
                                  [-5., -5., -5., -5., -5., -4., -5., -5., -5., -4., -5.],
                                  [-5., -5., -5., -5., -5., -5., -5., -5., -5., -4., -5.],
                                  [-6., -6., -6., -6., -6., -6., -6., -6., -6., -5., -6.]]]
        """
        assert gold_y.size() == sampled_y.size()
        b_sz, seq_len = gold_y.size()
        q_values = gold_y.new_zeros((b_sz, seq_len + 1, vocab_size), dtype=torch.float)
        edit_dists = gold_y.new_zeros((b_sz, seq_len + 1, seq_len + 1), dtype=torch.float)

        # run batch version of the levenshtein algorithm
        edit_dists[:, :, 0] = torch.arange(seq_len + 1)
        edit_dists[:, 0, :] = torch.arange(seq_len + 1)
        for i in range(1, seq_len + 1):
            for j in range(1, seq_len + 1):
                cost = (sampled_y[:, i-1] != gold_y[:, j-1]).float()
                min_cost, _ = torch.cat(((edit_dists[:, i-1, j] + 1).unsqueeze(dim=1),
                                         (edit_dists[:, i, j-1] + 1).unsqueeze(dim=1),
                                         (edit_dists[:, i-1, j-1] + cost).unsqueeze(dim=1)),
                                        dim=1).min(dim=1)
                edit_dists[:, i, j] = min_cost
        # #

        # find gold next tokens and update their QValues
        edit_dists_mask = OCD.edit_distance_mask(gold_y, end_symbol_id)
        edit_dists = edit_dists.masked_fill_(edit_dists_mask, LARGE_NUM)
        min_dist, _ = edit_dists.min(dim=2)
        min_dist = min_dist.unsqueeze(dim=2)
        steps_with_min_dists = (edit_dists == min_dist)
        extended_gold_y = gold_y.repeat(1, seq_len + 1).view(b_sz, seq_len + 1, seq_len)
        indices = steps_with_min_dists.nonzero()
        gold_next_tokens = extended_gold_y[indices.split(1, dim=1)]
        indices[:, 2] = gold_next_tokens.squeeze(dim=1)
        q_values[indices.split(1, dim=1)] = 1
        q_values = q_values - (1 + min_dist)
        return q_values[:, :-1, :]  # ignore the step 'seq_len + 1'

    @staticmethod
    def compute_optimal_pi(q_values: torch.FloatTensor,
                           tau: Optional[float] = None) -> torch.FloatTensor:
        """ computing the optimal policy
        Args:
            q_values (`~torch.FloatTensor`): ``(batch_size, sequence_length, vocab_size)``
            tau (float): normalizer parameter for softmax >= 0

        Returns:
            `~torch.FloatTensor`: ``(batch_size, sequence_length, vocab_size)``
                                  the optimal gold policy (distribution)
        """
        if tau == 0 or tau is None:
            # tau limits zero
            normalized_q_values = q_values * LARGE_NUM
        else:
            assert tau > 0
            normalized_q_values = q_values / tau

        assert len(normalized_q_values.size()) == 3
        return nn.functional.softmax(normalized_q_values, dim=2)

    @staticmethod
    def loss(optimal_pi: torch.FloatTensor,
             model_log_probs: torch.FloatTensor,
             sampled_y_mask: torch.ByteTensor) -> torch.Tensor:
        """ ocd loss
        Args:
            optimal_pi (`~torch.FloatTensor`): ``(batch_size, sequence_length, vocab_size)``
            model_log_probs (`~torch.FloatTensor`): ``(batch_size, sequence_length, vocab_size)``
                                                    log probabilities of each output token at each step.
            sampled_y_mask (`~torch.ByteTensor`): zero for actual steps,
                                                  ones for masked padded irrelevant steps.
        Return:
            `~torch.Tensor`: loss to backpropagate
        """
        loss = nn.functional.kl_div(model_log_probs, optimal_pi, reduction='none')
        loss = loss.sum(dim=2)
        # ignore steps after end_symbol
        loss = loss * (1 - sampled_y_mask).float()
        loss = loss.sum(dim=1)
        return loss.mean(dim=0)
