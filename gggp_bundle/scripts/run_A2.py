"""
gggp_bundle/scripts/run_A2.py

MEDP A2 / S2c -- evolutionary runner for the UMC (G, D) pair.

Scope shift from A1:
  * A1 optimised a single scalar F_raw = mean_i cos(T_i, D(G(T_i))).
  * A2 scalarizes four neural-constraint components:
        F_umc = w_nc4*F_nc4 + w_nc1*F_nc1 + w_nc2*F_nc2 + w_nc3*F_nc3
    where
        F_nc4 = mean cos(T_i, D(G(T_i)))                   (A1 carry-over)
        F_nc1 = mean cos(c_i, G(D(c_i)))                   (dual fixed point)
        F_nc2 = triplet-acc(D(0.5 c_i + 0.5 c_j) vs (T_i+T_j)/2 vs T_k)
        F_nc3 = fraction of D-opcodes that are CTRL / SBC / ADDC
    Penalties (compactness / class / seed-std) are identical to A1.
    See config/fitness.toml [fitness] + [nc_weights] and scripts/fitness.py.

Reads:
  config/ea.toml               EA hyperparameters incl. [a2] section
  config/fitness.toml          shaping weights incl. [nc_weights]
  demos/semiotic_hypercube/T_v2_pca.npy        (256, 16) corpus embeddings
  demos/semiotic_hypercube/classes_v2.npy      (256,)    ground-truth labels
  demos/semiotic_hypercube/grammar_encoder_a2.cfg        dim=code_dim=8
  demos/semiotic_hypercube/grammar_decoder_a2_nc3.cfg    decoder-nc3,
                                                         target=16 code=8

Writes:
  demos/semiotic_hypercube/runA2_seed<seed>.jsonl        per-gen log with
                                                         F_nc1..4 breakdown
  demos/semiotic_hypercube/runA2_seed<seed>_best.json    best individual +
                                                         full NC breakdown

EA loop (same skeleton as run_A1.py; only evaluation changes):
  1. Seed split of N rows into train / test (train_fraction).
  2. Init population of (G, D) pairs via SH.random_chromosome.
  3. Per generation:
       a. Sample one fixed set of NC2 pair indices for the whole gen
          (shared across all individuals so NC2 scores are comparable).
       b. For each individual:
            - batch_render_dual -> (c, T_hat, F_nc4)
            - render G over T_hat (loop) -> c_hat -> F_nc1
            - render D over (c_i + c_j)/2 (loop over pairs) -> T_mix
              F_nc2 = triplet_acc(T_mix, (T_i+T_j)/2, T_k)
            - chromosome_text(chromo_d, 'decoder') -> F_nc3
            - shape_fitness_umc -> F_umc + shaped F + breakdown
       c. Elitism + tournament + crossover + mutation (A1 pattern).
  4. After last generation: evaluate best individual on TEST split and
     log train/test gap for each NC component.

Run:
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/run_A2.py --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))
from fitness import (  # noqa: E402
    FitnessConfig,
    NCWeights,
    compute_F_nc1,
    compute_F_nc2,
    compute_F_nc3_signal,
    compute_F_nc4,
    shape_fitness_umc,
)

BUNDLE = REPO_ROOT / "gggp_bundle"
DEMO_DIR = BUNDLE / "demos" / "semiotic_hypercube"
EA_CONFIG = BUNDLE / "config" / "ea.toml"
FITNESS_CONFIG = BUNDLE / "config" / "fitness.toml"

# A2 PCA-only asset layout. RAW (1024-dim) is intentionally not
# supported: A1.1 already demonstrated that PCA-16 dominates raw-1024
# on F_nc4, and NC2/NC3 would be noisier at 1024.
A2_ASSETS = {
    "T_path":          DEMO_DIR / "T_v2_pca.npy",
    "classes_path":    DEMO_DIR / "classes_v2.npy",
    "encoder_grammar": DEMO_DIR / "grammar_encoder_a2.cfg",
    "decoder_grammar": DEMO_DIR / "grammar_decoder_a2_nc3.cfg",
}


# ================================================================ config ==
@dataclass
class A2Config:
    """Merged EA + A2-section config. Loaded from config/ea.toml."""

    pop_size: int
    n_generations: int
    tournament_k: int
    p_crossover: float
    p_mutation: float
    elitism: int
    train_fraction: float
    log_every: int
    max_gene_count: int
    max_resample_attempts: int
    code_dim: int
    target_dim: int
    nc2_pairs: int

    @classmethod
    def load(cls, path: Path = EA_CONFIG) -> "A2Config":
        with path.open("rb") as f:
            data = tomllib.load(f)
        ea = data["ea"]
        chromo = data["chromosome"]
        a2 = data.get("a2")
        if a2 is None:
            raise KeyError(
                f"{path}: missing [a2] section. Run medp A2/S2c landed it; "
                f"did you check out an A1-only revision?"
            )
        return cls(
            pop_size=int(ea["pop_size"]),
            n_generations=int(ea["n_generations"]),
            tournament_k=int(ea["tournament_k"]),
            p_crossover=float(ea["p_crossover"]),
            p_mutation=float(ea["p_mutation"]),
            elitism=int(ea["elitism"]),
            train_fraction=float(ea["train_fraction"]),
            log_every=int(ea["log_every"]),
            max_gene_count=int(chromo["max_gene_count"]),
            max_resample_attempts=int(chromo["max_resample_attempts"]),
            code_dim=int(a2["code_dim"]),
            target_dim=int(a2["target_dim"]),
            nc2_pairs=int(a2["nc2_pairs"]),
        )


# ============================================================== fixtures ==
def ensure_grammars(code_dim: int, target_dim: int) -> tuple[Path, Path]:
    """Build A2 grammars if missing.

    Encoder = custom dim=code_dim (baseline 6 ops).
    Decoder = decoder-nc3-custom (baseline 6 + CTRL/SBC/ADDC) at
              target_dim + code_dim bounds.
    """
    g_enc = A2_ASSETS["encoder_grammar"]
    g_dec = A2_ASSETS["decoder_grammar"]
    missing = [p for p in (g_enc, g_dec) if not p.is_file()]
    if not missing:
        return g_enc, g_dec
    print(f"[run_A2] regenerating grammars: {[str(m) for m in missing]}")
    rust_dir = BUNDLE / "rust"
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "gen_neuro_grammar"],
        cwd=rust_dir, check=True,
    )
    bin_path = rust_dir / "target" / "release" / "gen_neuro_grammar"
    if not g_enc.is_file():
        subprocess.run(
            [str(bin_path), "custom", str(code_dim), str(g_enc)],
            check=True,
        )
    if not g_dec.is_file():
        subprocess.run(
            [
                str(bin_path),
                "decoder-nc3-custom",
                str(target_dim),
                str(code_dim),
                str(g_dec),
            ],
            check=True,
        )
    return g_enc, g_dec


def load_corpus(cfg: A2Config) -> tuple[np.ndarray, np.ndarray]:
    """Load T_v2_pca + classes_v2. Raises SystemExit with a hint if
    either file is missing (S0b/S0c haven't been run yet).
    """
    t_path = A2_ASSETS["T_path"]
    c_path = A2_ASSETS["classes_path"]
    missing = [str(p) for p in (t_path, c_path) if not p.is_file()]
    if missing:
        raise SystemExit(
            f"A2 corpus missing: {missing}. "
            f"Run scripts/embed_corpus.py (S0b) on corpus_v2.jsonl and "
            f"scripts/pca_reduce.py (S0c) to produce T_v2_pca.npy / "
            f"classes_v2.npy before invoking run_A2.py."
        )
    T = np.load(t_path).astype(np.float64)
    classes = np.load(c_path)
    if T.shape[1] != cfg.target_dim:
        raise SystemExit(
            f"T_v2_pca shape {T.shape} inconsistent with "
            f"target_dim={cfg.target_dim} in ea.toml [a2]. "
            f"Re-run S0c with the correct target_dim."
        )
    if T.shape[0] != classes.shape[0]:
        raise SystemExit(
            f"T rows {T.shape[0]} != classes rows {classes.shape[0]}. "
            f"Corpus and labels are out of sync -- re-run S0b/S0c."
        )
    return np.ascontiguousarray(T), classes


def split_train_test(
    n: int, train_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    return idx[:n_train], idx[n_train:]


def sample_nc2_pairs(
    n_rows: int, m_pairs: int, rng: np.random.Generator
) -> np.ndarray:
    """Return (M, 3) int64 array of (i, j, k) indices for NC2 triplets.

    Guarantees i != j (pair anchors are distinct) and k != i and k != j
    (negative row is distinct from both anchors). With n_rows >= 3 this
    is always satisfiable; we assert it upfront so callers get a clear
    error rather than an infinite loop.
    """
    if n_rows < 3:
        raise ValueError(
            f"sample_nc2_pairs: n_rows={n_rows} < 3; cannot build triplets "
            f"with distinct i, j, k. Increase the training split size."
        )
    out = np.empty((m_pairs, 3), dtype=np.int64)
    # Sample with rejection per-triplet. m_pairs is O(N); cheap.
    for m in range(m_pairs):
        i, j = rng.choice(n_rows, size=2, replace=False)
        # Choose k uniformly from rows != i, j.
        while True:
            k = int(rng.integers(n_rows))
            if k != i and k != j:
                break
        out[m, 0] = i
        out[m, 1] = j
        out[m, 2] = k
    return out


# ================================================================ EA ops ==
@dataclass
class Individual:
    chromo_g: list[int]
    chromo_d: list[int]
    F_umc: float = float("nan")
    F_shaped: float = float("nan")
    F_nc1: float = float("nan")
    F_nc2: float = float("nan")
    F_nc3: float = float("nan")
    F_nc4: float = float("nan")
    breakdown: dict = field(default_factory=dict)

    def combined_len(self) -> int:
        return len(self.chromo_g) + len(self.chromo_d)


def tournament_select(
    pop: list[Individual], k: int, rng: random.Random
) -> Individual:
    picks = rng.sample(pop, min(k, len(pop)))
    return max(picks, key=lambda ind: ind.F_shaped)


def one_point_crossover(
    parent_a: list[int], parent_b: list[int], rng: random.Random
) -> tuple[list[int], list[int]]:
    if not parent_a or not parent_b:
        return list(parent_a), list(parent_b)
    cut_a = rng.randint(0, len(parent_a))
    cut_b = rng.randint(0, len(parent_b))
    child1 = parent_a[:cut_a] + parent_b[cut_b:]
    child2 = parent_b[:cut_b] + parent_a[cut_a:]
    return child1, child2


def mutate_inplace(
    chromo: list[int], p: float, max_val: int, rng: random.Random
) -> None:
    """Per-gene uniform mutation over [0, max_val].

    A1's max_val=15 stays safe for A2: the decoder-nc3 OP rule has 9
    choices, encoder OP rule has 6, other rules have <= 9, and numeric
    params run up to 15 (0..15 axis bounds). Offspring with any gene
    out of range are filtered by the validity gate downstream.
    """
    for i in range(len(chromo)):
        if rng.random() < p:
            chromo[i] = rng.randint(0, max_val)


def fresh_pair(
    sh: Any, rng: random.Random, cfg: A2Config
) -> tuple[list[int], list[int]]:
    """Generate a grammar-valid (G, D) chromosome pair via Rust seeds."""
    seed_g = rng.randint(0, 2**63 - 1)
    seed_d = rng.randint(0, 2**63 - 1)
    return (
        sh.random_chromosome(seed_g, "encoder"),
        sh.random_chromosome(seed_d, "decoder"),
    )


# ============================================================== evaluate ==
def evaluate_individual(
    sh: Any,
    ind: Individual,
    T_train: np.ndarray,
    classes_train: np.ndarray,
    pair_indices: np.ndarray,
    cfg: A2Config,
    fcfg: FitnessConfig,
    ncw: NCWeights,
) -> None:
    """Fill ind.F_* and ind.breakdown via the full NC1..NC4 pipeline.

    Failures at any stage degrade the individual to fallback fitness
    with an error reason in the breakdown; they do NOT crash the gen.
    """
    try:
        # --- NC4 + c, T_hat --------------------------------------------
        res = sh.batch_render_dual(
            ind.chromo_g, ind.chromo_d, T_train, cfg.code_dim, cfg.target_dim
        )
        F_nc4 = float(res["F"])
        c = np.asarray(res["c"], dtype=np.float64)
        T_hat = np.asarray(res["reconstruction"], dtype=np.float64)

        # --- NC1: c_hat_i = G(D(c_i)) == G(T_hat_i) --------------------
        # G is parsed against the encoder grammar (default role) and
        # T_hat_i is the input seed.
        n = T_hat.shape[0]
        c_hat = np.empty_like(c)
        for i in range(n):
            row = np.ascontiguousarray(T_hat[i])
            c_hat[i] = sh.render_tree_with_input(
                ind.chromo_g, cfg.code_dim, row
            )
        F_nc1 = compute_F_nc1(c, c_hat)

        # --- NC2: triplet over decoder-composed midpoints --------------
        i_idx = pair_indices[:, 0]
        j_idx = pair_indices[:, 1]
        k_idx = pair_indices[:, 2]
        m = pair_indices.shape[0]
        T_mix = np.empty((m, cfg.target_dim), dtype=np.float64)
        for p_ix in range(m):
            i = int(i_idx[p_ix])
            j = int(j_idx[p_ix])
            c_mix = 0.5 * (c[i] + c[j])
            c_mix = np.ascontiguousarray(c_mix)
            T_mix[p_ix] = sh.render_tree_with_input(
                ind.chromo_d, cfg.target_dim, c_mix, role="decoder"
            )
        T_avg = 0.5 * (T_train[i_idx] + T_train[j_idx])
        T_rand = T_train[k_idx]
        F_nc2 = compute_F_nc2(T_mix, T_avg, T_rand)

        # --- NC3: structural signal on decoder program text -----------
        d_text = sh.chromosome_text(ind.chromo_d, "decoder")
        F_nc3 = compute_F_nc3_signal(d_text)

        # --- Scalarize + penalize -------------------------------------
        F_shaped, bd = shape_fitness_umc(
            F_nc1=F_nc1, F_nc2=F_nc2, F_nc3=F_nc3, F_nc4=F_nc4,
            c_matrix=c, classes=classes_train,
            len_g=len(ind.chromo_g), len_d=len(ind.chromo_d),
            cfg=fcfg, nc_weights=ncw, seed_F_array=None,
        )
        ind.F_nc1 = F_nc1
        ind.F_nc2 = F_nc2
        ind.F_nc3 = F_nc3
        ind.F_nc4 = F_nc4
        ind.F_umc = bd.get("F_umc", float("nan"))
        ind.F_shaped = F_shaped
        ind.breakdown = bd
    except Exception as e:
        ind.F_nc1 = float("nan")
        ind.F_nc2 = float("nan")
        ind.F_nc3 = float("nan")
        ind.F_nc4 = float("nan")
        ind.F_umc = float("nan")
        ind.F_shaped = fcfg.fallback_fitness
        ind.breakdown = {"error": str(e)[:200]}


def evaluate_population(
    sh: Any,
    pop: list[Individual],
    T_train: np.ndarray,
    classes_train: np.ndarray,
    pair_indices: np.ndarray,
    cfg: A2Config,
    fcfg: FitnessConfig,
    ncw: NCWeights,
) -> None:
    for ind in pop:
        evaluate_individual(
            sh, ind, T_train, classes_train, pair_indices, cfg, fcfg, ncw
        )


# ================================================================ runner ==
def run(seed: int, out_jsonl: Path, out_best: Path) -> dict:
    cfg = A2Config.load()
    fcfg = FitnessConfig.load()
    ncw = NCWeights.load()
    g_enc, g_dec = ensure_grammars(cfg.code_dim, cfg.target_dim)

    T, classes = load_corpus(cfg)
    train_idx, test_idx = split_train_test(
        T.shape[0], cfg.train_fraction, seed=seed
    )
    T_train = np.ascontiguousarray(T[train_idx])
    classes_train = classes[train_idx]
    T_test = np.ascontiguousarray(T[test_idx])
    classes_test = classes[test_idx]
    print(
        f"[run_A2] seed={seed} N={T.shape[0]} "
        f"train={len(train_idx)} test={len(test_idx)} "
        f"target_dim={cfg.target_dim} code_dim={cfg.code_dim} "
        f"nc2_pairs={cfg.nc2_pairs}"
    )

    from semiotic_hypercube import SemioticHypercube
    sh = SemioticHypercube(str(g_enc))
    sh.attach_decoder_grammar(str(g_dec))

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # ---- Initial population -------------------------------------------
    pop: list[Individual] = []
    for _ in range(cfg.pop_size):
        g, d = fresh_pair(sh, py_rng, cfg)
        pop.append(Individual(chromo_g=g, chromo_d=d))

    # ---- Evolution ----------------------------------------------------
    t0 = time.time()
    history = []
    with out_jsonl.open("w", encoding="utf-8") as log_f:
        for gen in range(cfg.n_generations + 1):
            # Resample NC2 pair set deterministically per generation. All
            # individuals in this gen share the triplets so NC2 scores
            # are directly comparable (important for tournament
            # selection fairness).
            gen_rng = np.random.default_rng((seed * 10**6) + gen)
            pair_indices = sample_nc2_pairs(
                T_train.shape[0], cfg.nc2_pairs, gen_rng
            )
            evaluate_population(
                sh, pop, T_train, classes_train, pair_indices, cfg, fcfg, ncw
            )
            pop.sort(key=lambda x: x.F_shaped, reverse=True)

            best = pop[0]
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seed": seed,
                "gen": gen,
                "F_best_shaped": best.F_shaped,
                "F_best_umc": best.F_umc,
                "F_best_nc1": best.F_nc1,
                "F_best_nc2": best.F_nc2,
                "F_best_nc3": best.F_nc3,
                "F_best_nc4": best.F_nc4,
                "F_mean_shaped": float(
                    np.mean([i.F_shaped for i in pop])
                ),
                "best_len_g": len(best.chromo_g),
                "best_len_d": len(best.chromo_d),
                "silhouette_cos": best.breakdown.get("silhouette_cos"),
            }
            history.append(entry)
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_f.flush()
            if gen % cfg.log_every == 0 or gen == cfg.n_generations:
                print(
                    f"[run_A2] seed={seed} gen={gen:3d} "
                    f"F_shaped={best.F_shaped:+.4f} F_umc={best.F_umc:+.4f} "
                    f"NC4={best.F_nc4:+.3f} NC1={best.F_nc1:+.3f} "
                    f"NC2={best.F_nc2:+.3f} NC3={best.F_nc3:.3f} "
                    f"len=({len(best.chromo_g)},{len(best.chromo_d)})"
                )

            if gen == cfg.n_generations:
                break

            # ---- Build next generation -------------------------------
            next_pop: list[Individual] = [
                copy.deepcopy(pop[i]) for i in range(min(cfg.elitism, len(pop)))
            ]
            while len(next_pop) < cfg.pop_size:
                a = tournament_select(pop, cfg.tournament_k, py_rng)
                b = tournament_select(pop, cfg.tournament_k, py_rng)

                if py_rng.random() < cfg.p_crossover:
                    ch_g1, ch_g2 = one_point_crossover(
                        a.chromo_g, b.chromo_g, py_rng
                    )
                    ch_d1, ch_d2 = one_point_crossover(
                        a.chromo_d, b.chromo_d, py_rng
                    )
                else:
                    ch_g1, ch_g2 = list(a.chromo_g), list(b.chromo_g)
                    ch_d1, ch_d2 = list(a.chromo_d), list(b.chromo_d)

                for chromo in (ch_g1, ch_g2, ch_d1, ch_d2):
                    mutate_inplace(chromo, cfg.p_mutation, 15, py_rng)
                    while len(chromo) > cfg.max_gene_count:
                        chromo.pop()

                for offspring in (
                    Individual(ch_g1, ch_d1),
                    Individual(ch_g2, ch_d2),
                ):
                    # Validity gate: batch_render_dual on a tiny slice.
                    # If this passes, render_tree_with_input(role='decoder')
                    # and chromosome_text(role='decoder') will also pass --
                    # they share tree_from_chromosome.
                    valid = False
                    for _ in range(cfg.max_resample_attempts):
                        try:
                            sh.batch_render_dual(
                                offspring.chromo_g,
                                offspring.chromo_d,
                                T_train[:1],
                                cfg.code_dim,
                                cfg.target_dim,
                            )
                            valid = True
                            break
                        except Exception:
                            offspring.chromo_g = sh.random_chromosome(
                                py_rng.randint(0, 2**63 - 1), "encoder"
                            )
                            offspring.chromo_d = sh.random_chromosome(
                                py_rng.randint(0, 2**63 - 1), "decoder"
                            )
                    if not valid:
                        g, d = fresh_pair(sh, py_rng, cfg)
                        offspring.chromo_g = g
                        offspring.chromo_d = d
                    if len(next_pop) < cfg.pop_size:
                        next_pop.append(offspring)

            pop = next_pop

    wall = time.time() - t0
    print(f"[run_A2] seed={seed} wall={wall:.1f}s")

    # ---- Test-split generalization ------------------------------------
    pop.sort(key=lambda x: x.F_shaped, reverse=True)
    best = pop[0]
    test_rng = np.random.default_rng((seed * 10**6) + 10**9)  # disjoint from gen seeds
    test_pairs = sample_nc2_pairs(T_test.shape[0], cfg.nc2_pairs, test_rng)
    test_ind = Individual(chromo_g=best.chromo_g, chromo_d=best.chromo_d)
    evaluate_individual(
        sh, test_ind, T_test, classes_test, test_pairs, cfg, fcfg, ncw
    )
    print(
        f"[run_A2] seed={seed} TEST F_shaped={test_ind.F_shaped:+.4f} "
        f"NC4={test_ind.F_nc4:+.3f} NC1={test_ind.F_nc1:+.3f} "
        f"NC2={test_ind.F_nc2:+.3f} NC3={test_ind.F_nc3:.3f} "
        f"(train F_shaped={best.F_shaped:+.4f} "
        f"gap={best.F_shaped - test_ind.F_shaped:+.4f})"
    )

    out_best.write_text(
        json.dumps(
            {
                "seed": seed,
                "chromo_g": best.chromo_g,
                "chromo_d": best.chromo_d,
                "train": {
                    "F_shaped": best.F_shaped,
                    "F_umc": best.F_umc,
                    "F_nc1": best.F_nc1,
                    "F_nc2": best.F_nc2,
                    "F_nc3": best.F_nc3,
                    "F_nc4": best.F_nc4,
                    "breakdown": best.breakdown,
                },
                "test": {
                    "F_shaped": test_ind.F_shaped,
                    "F_umc": test_ind.F_umc,
                    "F_nc1": test_ind.F_nc1,
                    "F_nc2": test_ind.F_nc2,
                    "F_nc3": test_ind.F_nc3,
                    "F_nc4": test_ind.F_nc4,
                    "breakdown": test_ind.breakdown,
                },
                "wall_s": wall,
                "ea_config": {
                    "pop_size": cfg.pop_size,
                    "n_generations": cfg.n_generations,
                    "p_crossover": cfg.p_crossover,
                    "p_mutation": cfg.p_mutation,
                    "tournament_k": cfg.tournament_k,
                    "elitism": cfg.elitism,
                    "train_fraction": cfg.train_fraction,
                    "code_dim": cfg.code_dim,
                    "target_dim": cfg.target_dim,
                    "nc2_pairs": cfg.nc2_pairs,
                },
                "nc_weights": {
                    "w_nc1": ncw.w_nc1,
                    "w_nc2": ncw.w_nc2,
                    "w_nc3": ncw.w_nc3,
                    "w_nc4": ncw.w_nc4,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return {
        "seed": seed,
        "F_train_shaped": best.F_shaped,
        "F_test_shaped": test_ind.F_shaped,
        "wall_s": wall,
    }


# =================================================================== main ==
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_jsonl = DEMO_DIR / f"runA2_seed{args.seed}.jsonl"
    out_best = DEMO_DIR / f"runA2_seed{args.seed}_best.json"

    summary = run(args.seed, out_jsonl, out_best)
    print(f"[run_A2] summary: {summary}")
    print(f"[run_A2] wrote {out_jsonl}")
    print(f"[run_A2] wrote {out_best}")


if __name__ == "__main__":
    main()
