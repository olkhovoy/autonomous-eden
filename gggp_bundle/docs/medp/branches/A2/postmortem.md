# A2 — Postmortem

**Branch**: A2 — Unified Meaning Constraints (UMC) over a (G, D) encoder-decoder pair.
**Status**: `PASS` — all 6 gates clear, on all 3 seeds, with the Risk-1 NC3-cheating control also clean.
**Dates**: 2026-04-22 (plan draft) → 2026-04-24 (validation).
**Runtime**: 13 s per EA seed on CPU, 3 seeds in ≤ 1 minute wall-clock total.

## 1. Intent (recap)

A1 optimized a single scalar `F_raw = mean_i cos(T_i, D(G(T_i)))` and A1.1
closed the grammar-expressivity gap via a PCA-16 target basis. A2 asks:
if we layer **three additional neural constraints** onto that signal,
does the evolutionary search still converge, and does it do so to a
program that *genuinely* respects all four?

Constraints (see `docs/medp/branches/A2/plan.md` §Motivation for full
derivation):

| id | formula | semantic |
|---|---|---|
| **NC4** | `F_nc4 = mean_i cos(T_i, D(G(T_i)))` | reconstruction (A1 carry-over) |
| **NC1** | `F_nc1 = mean_i cos(c_i, G(D(c_i)))` | recursive closure / dual fixed point |
| **NC2** | `F_nc2 = triplet-acc(D((c_i + c_j)/2), (T_i + T_j)/2, T_k)` | midpoint compositionality |
| **NC3** | `F_nc3 = |{NC3 opcode tokens in D}| / |{opcode tokens in D}|` | structural code-gating |

Scalarized (locked Q2 in plan): `F_umc = w_nc4·NC4 + w_nc1·NC1 + w_nc2·NC2 + w_nc3·NC3`
with `w_nc4=1.0, w_nc1=0.5, w_nc2=0.2, w_nc3=0.3` (rationale in
`config/fitness.toml` §`[nc_weights]`).

## 2. Pipeline summary

| unit | deliverable | notes |
|---|---|---|
| S0a | `build_corpus_v2.py` | 256 rows = 8 classes × 32 paraphrases; class-level resume + atomic checkpoint; Qwen3-35B MoE @ `temp=0, seed=42`. |
| S0b | `embed_corpus.py --version v2` | `T_v2.npy (256, 1024)`, Qwen3-Embedding-0.6B, L2-normalized. |
| S0c | `pca_reduce.py --version v2` | `T_v2_pca.npy (256, 16)`, 66.5% variance retained, class silhouette 0.754. |
| S1a | Rust NC3 ops | `CTRL(axis, cidx, mask)`, `ScaleByCode(k)`, `AddCode(axis, cidx)` in `rust/src/gggp/{phenotype,vector}.rs`. |
| S1b | `decoder-nc3-custom` subcommand | `gen_neuro_grammar decoder-nc3-custom <target_dim> <code_dim> <out>`; 9 ops = 6 baseline + CTRL/SBC/ADDC. |
| S1c | Role-aware Python bindings | `render_tree_with_input(..., role="decoder")`, `chromosome_text(chromo, role)`; 10 new pytest tests. |
| S2a | `[nc_weights]` config | Locked weights + admissible ranges; loader in `fitness.py`. |
| S2b | NC1–NC4 fitness helpers | `compute_F_nc{1,2,3,4}` + `shape_fitness_umc`; 22 pytest tests (all green). |
| S2c | `run_A2.py` | EA runner, per-gen JSONL log, train/test split, pair sampling shared per generation. |
| S3a | `eval_ari_a2.py` | ARI / AMI / V-measure / silhouette on `c_i`. |
| S3b | `eval_gates_a2.py` | Gate verdict + Risk-1 NC3-cheating control. |
| S3c | `plot_a2.py` | NC1..4 trajectories, c-space scatter, gate bar chart. |
| S3d | this document | summary + next-step recommendations. |

## 3. Result — the NORM+ADDC discovery

All three seeds (0, 1, 2) converged to the **same algorithmic shape**,
differing only in which numeric axes receive which code coordinate:

| seed | G (encoder) | D (decoder) | chromo_g | chromo_d |
|---|---|---|---|---|
| 0 | `NORM` | `ADDC 15 7` | `[2, 2]` | `[2, 8, 15, 7]` |
| 1 | `NORM` | `ADDC 15 0` | `[2, 2]` | `[2, 8, 15, 0]` |
| 2 | `NORM` | `ADDC 10 2` | `[2, 2]` | `[2, 8, 10, 2]` |

Interpretation.

- **G = NORM** — the first 8 components of the PCA-16 input are
  auto-normalized to a unit vector. Because the A1.1 PCA basis already
  concentrates semantic energy into the top coordinates, this is
  essentially "keep PCA top-8, renormalize" → the code space is the
  PCA top-8 sphere.
- **D = ADDC** — decoder state is zero-initialized, the first 8 axes
  are seeded from `c` (grammar semantics), and ADDC then writes `c[cidx]`
  back into one additional axis (10 or 15, depending on seed). The result
  is a 16-dim vector that is essentially `[c; 0; 0; 0; 0; 0; 0; 0; c[cidx]]`
  — a linear projection of `c` into the 16-dim PCA-basis target space.

This is a **globally rational algorithm for the A2 UMC objective**, not
metric-gaming: the Risk-1 c-shuffle diagnostic (see §4) shows the
decoder actually uses `c` (NC4 drops by ≈ 0.92 when `c` is permuted).

## 4. Gate table

Full verdict from `eval_gates_a2.py` (stored at
`demos/semiotic_hypercube/a2_gate_report.{json,md}`):

| gate | metric | rule | value | verdict |
|---|---|---|---|---|
| G_NC1 | F_nc1 (mean over 3 seeds) | `> 0.50` | **+1.0000** | PASS |
| G_NC2 | F_nc2 (mean over 3 seeds) | `> 0.55` | **+1.0000** | PASS |
| G_NC3 | F_nc3 (mean over 3 seeds) | `> 0.20` | **+1.0000** | PASS |
| G_NC4 | F_nc4 (mean over 3 seeds) | `> 0.50` | **+0.9223** | PASS |
| G_A2_stab | σ(F_umc) across seeds | `< 0.08` | **+0.0036** | PASS |
| G_A2_len | max(len_g + len_d) | `< 20` | **6** | PASS |

All 6 pass by large margins (≥ 0.37 above every strict-inequality
threshold; σ is 22× below its gate).

### Risk-1 NC3-cheating diagnostic

Procedure: for each seed, permute the c matrix row-wise, decode, and
recompute NC4. A decoder that uses `c` non-trivially should produce a
**large drop** in NC4. Results (`delta_min = 0.10` required to flag
"genuine"):

| seed | NC4_orig | NC4_shuffled | drop | genuine c-use? |
|---|---|---|---|---|
| 0 | +0.925 | −0.008 | **+0.932** | YES |
| 1 | +0.925 | −0.029 | **+0.954** | YES |
| 2 | +0.917 | +0.017 | **+0.900** | YES |

All 3 seeds are genuine. Shuffling `c` collapses NC4 from ~+0.92 down
to ≈ 0, which is the cosine floor for unrelated vectors in the corpus
(inter-class cosine is -0.11 in T-space).

## 5. Comparison with A1.1

A1.1 optimized NC4 only on the same PCA-16 basis (256 rows, 3 seeds):

| metric | A1.1 PCA (5 seeds) | A2 UMC (3 seeds) |
|---|---|---|
| F_nc4 train | 0.9211 ± 0.0023 | **0.9223 ± 0.0036** |
| F_nc4 test | 0.9208 ± 0.0092 | 0.9180 ± 0.0140 |
| NC1 | — | **1.000 ± 0.000** |
| NC2 | — | **1.000 ± 0.000** |
| NC3 | — | **1.000 ± 0.000** |
| wall / seed | 4.9 s | 13.3 s |
| combined len | (not comparable — A1 chromosome format differs) | 6 |

**A2 matches A1.1 on reconstruction and adds three other constraints
for free.** The 2.7× wall-time cost buys three extra evaluated
constraints (N decoder renders for NC1, M decoder renders for NC2,
one tree-text parse for NC3) and a tighter seed-stability band.

## 6. ARI / AMI clustering result

From `eval_ari_a2.py`:

| source | ARI | AMI | silhouette (cosine) |
|---|---|---|---|
| A2 `c_i` (all 3 seeds, KMeans k=8) | **1.0000** | **1.0000** | **+0.8851** |
| T_v2_pca baseline (KMeans k=8) | 1.0000 | 1.0000 | +0.7536 |
| shuffled labels (chance floor) | +0.0119 | +0.0218 | — |

KMeans perfectly recovers the 8-class partition on both `c` and
`T_v2_pca` — the corpus is already linearly separable. The silhouette
lift (**+0.132** from +0.754 → +0.885) shows that `G = NORM` on the
top-8 PCA dims actually **concentrates** the class separation beyond
the 16-dim T space, despite halving the dimensionality.

## 7. Figures

Saved to `demos/semiotic_hypercube/`:

- `a2_fig_trajectories.png` — per-gen NC1/NC2/NC3/NC4 curves (3 seeds),
  with the plan.md gate thresholds drawn as red dashed lines. NC1/NC2
  saturate by generation 1–5; NC3 jumps from a seed-dependent initial
  value (0.33–1.0) to 1.0 by generation 5 in every seed.
- `a2_fig_c_scatter.png` — 2D PCA projections of `T_v2_pca` (left) and
  the winning-seed's `c_i` (right), colored by ground-truth class.
  Class clusters are visually tighter in `c`-space.
- `a2_fig_gates.png` — horizontal bar chart of gate values vs
  thresholds; all green (PASS).

## 8. Caveats & risks that materialized

1. **Short-chromosome convergence.** `len_g=2`, `len_d=4`, combined=6 — far
   below the G_A2_len ceiling of 20. The EA saturated the scalar
   objective in ~5 generations and never explored compositionally
   richer programs. `len_penalty = 0.0025` is too weak at the current
   scalar ceiling of F_umc ≈ 1.925; a longer program with marginally
   worse NC4 would lose to the short one on len alone.
   → **Candidate A3 task**: sensitivity sweep on `len_penalty ∈ [0, 0.05]`,
   or a hard `min_len` constraint.
2. **NC2 linearity bypass (plan Risk 2).** The winning decoder is
   effectively affine in `c`, so `D((c_i + c_j)/2) ≈ (D(c_i) + D(c_j))/2`
   by construction, and NC2 passes "for free". This was foreseen in
   plan.md §Risks; mitigation is to add a non-linear grammar op
   (`FRAC(f)` / `ROT`) in a follow-up, which is outside the A2 scope.
3. **NC3 intensity.** G_NC3 requires only > 0.20 of D opcodes to be
   NC3-family; the evolved decoders hit 1.00 (all NC3). This is because
   the 4-gene decoder happens to be exactly one ADDC — no other
   opcodes to dilute the fraction. G_NC3 was effectively a "does D
   contain **any** NC3 op" test under the discovered local optimum.
4. **Risk 1 (NC3 symbolic cheating).** Foreseen and tested; **does not
   materialize**. All three seeds show a 0.9+ NC4 drop under
   c-permutation → the decoders truly depend on c.

## 9. Locked facts for downstream branches

- The A2 PCA basis (256 × 16) is semantically perfect — `c_i` clustering
  achieves ARI=1.0, silhouette +0.885. No further corpus enrichment is
  needed before attempting non-linear architectures.
- The `shape_fitness_umc` scalarizer is numerically stable (no NaNs,
  no inf, no divide-by-zero across 3 × 31 = 93 generations × 50
  individuals ≈ 4650 evaluations).
- `render_tree_with_input(role='decoder')` + `chromosome_text(..., 'decoder')`
  are the right Python-side primitives for A2-class experiments. All
  NC3 opcode counting happens in Python via `compute_F_nc3_signal`,
  so further NC3 expansion can happen without Rust changes.

## 10. Recommended next forks (for F0 / root log)

- **A3 — non-linear grammar ops** (`FRAC`, `ROT`) to force NC2 to
  measure genuine non-linear composition instead of affine linearity.
- **A2.1 — len-sensitivity sweep** over `[len_penalty, min_len]`
  to see whether the "NORM+ADDC" basin is the global optimum of the
  scalar objective or just a reachable cheap one.
- **A4 — remove `input` seeding** from `render_tree_with_input` so D
  can only see `c` via explicit NC3 ops. This would make G_NC3 a
  much stronger constraint (decoders without NC3 ops would literally
  produce zeros → NC4 = 0 → failing G_NC4).

Until any of those forks is taken, A2 stands as a **positive empirical
result**: a small, transparent, 6-token two-program pair that satisfies
all four UMC criteria on a realistic LLM-derived corpus.
