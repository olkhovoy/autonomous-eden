# Neurobars Autoresearch

This directory is a constrained experiment loop for improving the neurobar
encoder. The workflow is intentionally modeled after Karpathy's
`autoresearch`, but adapted to time-series representation learning.

## In Scope

Read these files before doing any experiment:

1. `README.md`
2. `prepare.py`
3. `train.py`

## What You Can Modify

- `train.py` only

## What You Must Not Modify

- `prepare.py`
- the evaluation metric
- the temporal split

## Objective

Minimize `val_score` on the fixed validation slice.

The score is:

`val_score = val_weighted_mse + 0.5 * val_close_delta_mse + 0.05 * (1 - val_direction_acc)`

Lower is better.

## Fixed Rules

1. Data prep is temporal, not random.
2. Normalization statistics are fit on the training split only.
3. Validation always starts at `2025-04-24 12:20:00 UTC`.
4. Training runs for a fixed wall-clock budget from `prepare.py`.
5. Keep the model interface: `model(x) -> (latent, next_bar_pred)`.
6. Do not add dependencies.

## Experiment Loop

1. Create `results.tsv` if missing with the header:

`commit	val_score	status	description`

2. Run the baseline first:

`python experiments/neurobars_autoresearch/train.py > run.log 2>&1`

3. Extract the summary:

`grep "^val_score:\|^val_weighted_mse:\|^val_close_delta_mse:\|^val_direction_acc:" run.log`

4. Keep only changes that improve `val_score`.
5. Prefer simple changes over complicated ones if the gain is small.

## Good Ideas

- architecture changes inside `train.py`
- better training loss weighting
- optimizer and scheduler changes
- batch size and hidden width changes
- regularization and normalization changes

## Bad Ideas

- making validation easier
- using future data in normalization
- changing the fixed metric
- changing the fixed split
