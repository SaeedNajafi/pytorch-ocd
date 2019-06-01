"""
tests for ocd functions.
"""
# pylint: disable=E1101
# pylint: disable=unused-argument
# pylint: disable=unused-variable
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import random
import torch
import pytest
import numpy as np
from ocd import OCD

@pytest.fixture(scope='module')
def fix_seed():
    seed = len('ocd testing')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.fixture(scope='module')
def expected_edit_dist_mask():
    mask = [[[0, 0, 0, 0, 1],
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
    return mask

@pytest.fixture(scope='module')
def expected_q_values():
    q_values = [[[0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                 [-1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                 [-2.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                 [-3.0, -2.0, -2.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
                 [-3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
                 [-4.0, -4.0, -3.0, -3.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
                 [-4.0, -4.0, -4.0, -4.0, -3.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
                 [-4.0, -4.0, -4.0, -4.0, -4.0, -3.0, -4.0, -4.0, -4.0, -4.0, -4.0],
                 [-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -3.0, -4.0]],
                [[0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                 [-1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                 [-2.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                 [-3.0, -2.0, -2.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
                 [-4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
                 [-4.0, -4.0, -4.0, -4.0, -4.0, -3.0, -4.0, -4.0, -4.0, -4.0, -4.0],
                 [-5.0, -5.0, -5.0, -5.0, -5.0, -4.0, -5.0, -5.0, -5.0, -4.0, -5.0],
                 [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -4.0, -5.0],
                 [-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -5.0, -6.0]]]
    return q_values

@pytest.fixture(scope='module')
def expected_policy():
    policy = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.50, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.3333, 0.3333, 0.3333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.250, 0.250, 0.250, 0.250, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 0.50, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    return policy

def test__mask(fix_seed, expected_edit_dist_mask):
    seq = [[1, 2, 3, 4],
           [5, 4, 6, 4],
           [3, 2, 4, 0]]
    end_symbol_id = 4
    seq = torch.LongTensor(seq)
    mask = OCD.sequence_mask(seq, end_symbol_id)
    expected_mask = [[0, 0, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]
    assert mask.tolist() == expected_mask
    edit_dist_mask = OCD.edit_distance_mask(seq, end_symbol_id)
    assert edit_dist_mask.tolist() == expected_edit_dist_mask

def test_edit_q_values(fix_seed, expected_q_values):
    """
    vocabulary = {'S':0, 'U':1, 'N':2, 'D':3,  'A':4 , 'Y':5,
                  'T':6, 'R':7, 'P':8, '</s>': 9, '<pad>': 10}
    """
    vocab_size = 11
    end_symbol_id = 9

    # batch_y = {'SUNDAY</s><pad><pad>', 'SUNDAY</s><pad><pad>'}
    gold_y = [[0, 1, 2, 3, 4, 5, 9, 10, 10],
              [0, 1, 2, 3, 4, 5, 9, 10, 10]]

    sampled = {'SATURDAY</s>', 'SATRAPY</s>U'}
    sampled_y = [[0, 4, 6, 1, 7, 3, 4, 5, 9],
                 [0, 4, 6, 7, 4, 8, 5, 9, 1]]
    q_values = OCD.edit_distance_q_values(torch.LongTensor(sampled_y),
                                          torch.LongTensor(gold_y),
                                          end_symbol_id, vocab_size)
    assert q_values.tolist() == expected_q_values

def test_policy(fix_seed, expected_q_values, expected_policy):
    optimal_pi = OCD.compute_optimal_pi(torch.FloatTensor(expected_q_values))
    assert (np.round(optimal_pi.tolist()[1], 4) == expected_policy).all()

def test_loss(fix_seed, expected_q_values):
    optimal_pi = OCD.compute_optimal_pi(torch.FloatTensor(expected_q_values))
    model_log_probs = torch.nn.functional.log_softmax(torch.randn_like(optimal_pi), dim=2)
    b_sz, seq_len, vocab_size = model_log_probs.size()
    sampled_y_mask = torch.zeros(b_sz, seq_len).byte()
    loss = OCD.loss(optimal_pi, model_log_probs, sampled_y_mask)
    assert round(loss.item(), 4) == 20.7344

def test_forward(fix_seed, expected_q_values):
    optimal_pi = OCD.compute_optimal_pi(torch.FloatTensor(expected_q_values))
    model_scores = torch.randn_like(optimal_pi)
    gold_y = [[0, 1, 2, 3, 4, 5, 9, 10, 10],
              [0, 1, 2, 3, 4, 5, 9, 10, 10]]
    ocd = OCD(vocab_size=11, end_symbol_id=9)
    loss = ocd(model_scores, torch.LongTensor(gold_y))
    assert round(loss.item(), 4) == 8.0762
