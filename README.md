# Optimal Completion Distillation (OCD) Training
Implementation of the Optimal Completion Distillation for Sequence Labeling </br>
source : https://arxiv.org/abs/1810.01398

## Requirements
`python3`, `pytorch 1.0.0`
</br>

[![CircleCI](https://circleci.com/gh/SaeedNajafi/pytorch-ocd/tree/master.svg?style=svg)](https://circleci.com/gh/SaeedNajafi/pytorch-ocd/tree/master)

## Install
```sh
virtualenv env
source env/bin/activate
pip3 install .
```
## How to use?
look at https://github.com/SaeedNajafi/pytorch-ocd/blob/master/ocd/__init__.py#L50
and </br>
https://github.com/SaeedNajafi/pytorch-ocd/blob/master/tests/test_ocd.py#L132
```python
from ocd import OCD

ocd_trainer = OCD(vocab_size=10, end_symbol_id=9)
...  # model defines scores for each step and each possible output token.
ocd_loss = ocd_trainer(model_scores, gold_output_sequence)
...  # backprop with ocd_loss
```
