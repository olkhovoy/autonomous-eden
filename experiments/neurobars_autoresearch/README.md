# Neurobars Autoresearch

This is an `autoresearch`-style loop for improving the neurobar encoder
without letting the agent rewrite data prep or the evaluation harness.

## Files

- `prepare.py` - fixed data prep, temporal split, normalization, dataloader, evaluation
- `train.py` - the only file the agent should modify during experiments
- `program.md` - instructions for the coding agent

## Quick Start

```bash
python experiments/neurobars_autoresearch/prepare.py
python experiments/neurobars_autoresearch/train.py
```

The fixed metric is `val_score`:

`val_score = val_weighted_mse + 0.5 * val_close_delta_mse + 0.05 * (1 - val_direction_acc)`

Lower is better.
