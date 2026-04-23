# Monolith Walk-Forward Probe

Last updated: 2026-03-10

## Purpose

This document records a preliminary answer to one question:

Can the current single-policy engine find candidates that stay profitable for
at least some time immediately after the in-sample window?

This is not a full validation of the engine.
It is only a walk-forward viability probe.

## Configuration

Script:

1. [probe_monolith_walkforward.py](/home/user/mcs/scripts/probe_monolith_walkforward.py)

Artifacts:

1. [summary.json](/home/user/mcs/checkpoints/monolith_walkforward_probe/summary.json)
2. [wf_01_20240101_run01.log](/home/user/mcs/checkpoints/monolith_walkforward_probe/wf_01_20240101_run01.log)
3. [wf_01_20240101_run02.log](/home/user/mcs/checkpoints/monolith_walkforward_probe/wf_01_20240101_run02.log)
4. [wf_02_20240601_run01.log](/home/user/mcs/checkpoints/monolith_walkforward_probe/wf_02_20240601_run01.log)
5. [wf_02_20240601_run02.log](/home/user/mcs/checkpoints/monolith_walkforward_probe/wf_02_20240601_run02.log)
6. [wf_03_20250501_run01.log](/home/user/mcs/checkpoints/monolith_walkforward_probe/wf_03_20250501_run01.log)
7. [wf_03_20250501_run02.log](/home/user/mcs/checkpoints/monolith_walkforward_probe/wf_03_20250501_run02.log)

Dataset:

1. [BTCUSDT_parquet_neurobars_autoresearch.npz](/home/user/mcs/data/BTCUSDT_parquet_neurobars_autoresearch.npz)

Economics:

1. `binance`
2. `10000 USD`
3. `fraction_of_equity`
4. `position_notional_fraction = 1.0`
5. `execution_fee_mode = taker`

Search budget per candidate:

1. `8` generations
2. population `8`
3. current `evolve_monolith.py` fitness unchanged

Walk-forward layout:

1. `7d train -> adjacent 7d OOS`
2. `2` independent searches per window
3. windows:
   `2024-01-01`,
   `2024-06-01`,
   `2025-05-01`

## Aggregate Result

Across `6` runs:

1. `5/6` candidates were profitable in-sample
2. `1/6` candidates were profitable on the immediately adjacent OOS window
3. `0/6` candidates beat the best directional baseline on OOS
4. median train PnL was about `+227`
5. median OOS PnL was about `-749`

Interpretation:

1. the engine can find train-positive local candidates
2. but adjacent-OOS persistence is weak
3. current hit rate is not high enough to treat the engine as already
   confirmed

## Window-Level Outcome

### 2024-01-01 Window

Results:

1. both runs were train-positive
2. both runs failed immediately on adjacent OOS
3. losses were large and came with many trades

Interpretation:

1. this looks like ordinary overfitting, not candidate persistence

### 2024-06-01 Window

Results:

1. one run was almost flat on train and slightly negative on OOS
2. one run was strongly positive on train and strongly negative on OOS

Interpretation:

1. this looks like defensive collapse at best, overfitting at worst

### 2025-05-01 Window

Results:

1. one run failed on OOS
2. one run stayed profitable on adjacent OOS

Best observed candidate:

1. train roughly `+145`
2. OOS roughly `+273`
3. OOS trades `40`
4. OOS win rate about `52.5%`

Important limitation:

1. even this best run did not beat the best directional baseline on the same
   OOS slice

Interpretation:

1. this is evidence that the engine can occasionally produce a living adjacent
   OOS candidate
2. it is not evidence that the engine is already good enough as a production
   candidate factory

## Verdict

Current status:

1. weak positive on engine viability
2. not a confirmation of adequacy

More explicitly:

1. the current engine is capable of finding train-positive systems
2. it can occasionally find a candidate that survives the next OOS window
3. but the adjacent-OOS success rate is too low to declare the engine
   validated

## Practical Consequence

Do not discard the engine yet.
But do not promote it as the final candidate generator either.

The next rational step is:

1. keep it as a provisional candidate engine
2. move on to candidate registry and trade-level evaluation infrastructure
3. re-test it later under a broader library-of-candidates workflow

## Naming Note

The term `monolith` is now misleading.
The current object is better understood as:

1. `candidate engine`
2. `candidate cell`
3. `single-policy search engine`

It is no longer the architectural target.
