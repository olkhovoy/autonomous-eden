"""
gggp_bundle/scripts/run_A1.py

MEDP A1 / T7 -- evolutionary runner for the shared-genome (G, D) pair.

Reads:
  config/ea.toml               EA hyperparameters (pop, n_gen, etc.)
  config/fitness.toml          shaping weights (alpha_len, beta_class, ...)
  demos/semiotic_hypercube/T.npy          (128, 1024) corpus embeddings
  demos/semiotic_hypercube/classes.npy    (128,)      ground-truth labels
  demos/semiotic_hypercube/grammar_encoder.cfg   (dim=16 G grammar)
  demos/semiotic_hypercube/grammar_decoder.cfg   (dim=1024 D grammar)

Writes:
  demos/semiotic_hypercube/runA1_seed<seed>.jsonl    per-generation log
  demos/semiotic_hypercube/runA1_seed<seed>_best.json best individual

EA loop (Python-driven; Rust is only used for render + grammar parsing):
  1. Seed split of 128 T_i into 80/20 (train/test); only train feeds fitness.
  2. Init population of POP (G, D) pairs via SH.random_chromosome.
  3. For each generation:
       * Evaluate fitness for every individual on train split via
         batch_render_dual + shape_fitness.
       * Elitism: copy top-ELITISM unchanged.
       * Fill the rest of next-gen via tournament-select + 1pt crossover
         + per-gene mutation. Grammar-invalid offspring trigger resample
         (up to max_resample_attempts; then fall back to fresh random).
  4. After the last generation, also evaluate F on the TEST split for
     the best individual (generalization sanity check, recorded in
     log.jsonl but NOT used to select).

Run:
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/run_A1.py --seed 0
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
from fitness import FitnessConfig, shape_fitness  # noqa: E402

BUNDLE = REPO_ROOT / "gggp_bundle"
DEMO_DIR = BUNDLE / "demos" / "semiotic_hypercube"
EA_CONFIG = BUNDLE / "config" / "ea.toml"
FITNESS_CONFIG = BUNDLE / "config" / "fitness.toml"
CLASSES_PATH = DEMO_DIR / "classes.npy"

# Mode-specific asset paths. `raw` = A1 original, `pca` = A1.1 refinement.
MODE_ASSETS = {
    "raw": {
        "T_path": DEMO_DIR / "T.npy",
        "encoder_grammar": DEMO_DIR / "grammar_encoder.cfg",
        "decoder_grammar": DEMO_DIR / "grammar_decoder.cfg",
        "code_dim_default": 16,
        "target_dim_default": 1024,
    },
    "pca": {
        "T_path": DEMO_DIR / "T_pca.npy",
        "encoder_grammar": DEMO_DIR / "grammar_encoder_pca.cfg",
        "decoder_grammar": DEMO_DIR / "grammar_decoder_pca.cfg",
        "code_dim_default": 8,
        "target_dim_default": 16,
    },
}


# ------------------------------------------------------------------ config --
@dataclass
class EaConfig:
    pop_size: int
    n_generations: int
    tournament_k: int
    p_crossover: float
    p_mutation: float
    elitism: int
    train_fraction: float
    log_every: int
    code_dim: int
    target_dim: int
    max_gene_count: int
    max_resample_attempts: int

    @classmethod
    def load(cls, mode: str, path: Path = EA_CONFIG) -> "EaConfig":
        """Load EA config. Mode-dependent defaults come from MODE_ASSETS
        but are overridden by ea.toml [dims] if explicitly set there.
        """
        if mode not in MODE_ASSETS:
            raise ValueError(
                f"Unknown run mode '{mode}'. Valid: {list(MODE_ASSETS.keys())}"
            )
        with path.open("rb") as f:
            data = tomllib.load(f)
        ea = data["ea"]
        chromo = data["chromosome"]
        assets = MODE_ASSETS[mode]
        dims = data.get("dims", {})
        # For raw mode use ea.toml [dims] verbatim; for pca mode override
        # with mode defaults (ea.toml's [dims] block was authored for raw).
        if mode == "pca":
            code_dim = assets["code_dim_default"]
            target_dim = assets["target_dim_default"]
        else:
            code_dim = int(dims.get("code_dim", assets["code_dim_default"]))
            target_dim = int(dims.get("target_dim", assets["target_dim_default"]))
        return cls(
            pop_size=int(ea["pop_size"]),
            n_generations=int(ea["n_generations"]),
            tournament_k=int(ea["tournament_k"]),
            p_crossover=float(ea["p_crossover"]),
            p_mutation=float(ea["p_mutation"]),
            elitism=int(ea["elitism"]),
            train_fraction=float(ea["train_fraction"]),
            log_every=int(ea["log_every"]),
            code_dim=code_dim,
            target_dim=target_dim,
            max_gene_count=int(chromo["max_gene_count"]),
            max_resample_attempts=int(chromo["max_resample_attempts"]),
        )


# ------------------------------------------------------------ fixtures ----
def ensure_grammars(mode: str, code_dim: int, target_dim: int) -> tuple[Path, Path]:
    assets = MODE_ASSETS[mode]
    g_enc, g_dec = assets["encoder_grammar"], assets["decoder_grammar"]
    missing = [p for p in (g_enc, g_dec) if not p.is_file()]
    if not missing:
        return g_enc, g_dec
    print(f"[run_A1] regenerating grammars: {[str(m) for m in missing]}")
    rust_dir = BUNDLE / "rust"
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "gen_neuro_grammar"],
        cwd=rust_dir, check=True,
    )
    bin_path = rust_dir / "target" / "release" / "gen_neuro_grammar"
    # raw mode uses role aliases; pca mode uses custom dim.
    if mode == "raw":
        subprocess.run([str(bin_path), "encoder", str(g_enc)], check=True)
        subprocess.run([str(bin_path), "decoder", str(g_dec)], check=True)
    else:
        subprocess.run(
            [str(bin_path), "custom", str(code_dim), str(g_enc)], check=True
        )
        subprocess.run(
            [str(bin_path), "custom", str(target_dim), str(g_dec)], check=True
        )
    return g_enc, g_dec


def split_train_test(
    n: int, train_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    return idx[:n_train], idx[n_train:]


# -------------------------------------------------------------- EA ops ----
@dataclass
class Individual:
    chromo_g: list[int]
    chromo_d: list[int]
    F_raw: float = float("nan")
    F_shaped: float = float("nan")
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

    max_val is an upper bound across grammar rules; invalid offspring
    are filtered by try_parse upstream. With the A1 grammar (6 OP
    choices, 3 SEQ choices, numeric params up to ~20), max_val=15 is
    safe for most positions.
    """
    for i in range(len(chromo)):
        if rng.random() < p:
            chromo[i] = rng.randint(0, max_val)


def try_parse(
    sh: Any, chromo_g: list[int], chromo_d: list[int]
) -> bool:
    """Quick validity check: render into a trivial (dim=1, input=None)
    state. If either grammar rejects the chromosome, Rust raises
    ValueError and we return False. Cheap: ~20 microseconds per pair.
    """
    try:
        sh.render_tree_with_input(chromo_g, 1, None)
    except Exception:
        return False
    try:
        # decoder must be parseable against its own grammar, which lives
        # on the SemioticHypercube's decoder_cfg. render_tree_with_input
        # uses self.cfg (encoder) so we verify D chromosome separately
        # by calling batch_render_dual with a tiny (1,1) matrix is too
        # heavyweight -- instead we use sh._try_parse_decoder if it
        # exists, else accept the chromosome and catch errors at the
        # real batch_render_dual call later.
        #
        # For now: accept by default, fall back to random-regen on
        # batch_render_dual failure.
        pass
    except Exception:
        return False
    return True


def fresh_pair(sh: Any, rng: random.Random, code_dim: int, target_dim: int
               ) -> tuple[list[int], list[int]]:
    """Generate a grammar-valid (G, D) chromosome pair via Rust seeds."""
    seed_g = rng.randint(0, 2**63 - 1)
    seed_d = rng.randint(0, 2**63 - 1)
    return (
        sh.random_chromosome(seed_g, "encoder"),
        sh.random_chromosome(seed_d, "decoder"),
    )


# ------------------------------------------------------------ fitness -----
def evaluate_population(
    sh: Any,
    pop: list[Individual],
    T_train: np.ndarray,
    classes_train: np.ndarray,
    cfg: EaConfig,
    fitness_cfg: FitnessConfig,
) -> None:
    """Fill in F_raw, F_shaped, breakdown on every individual in-place."""
    for ind in pop:
        try:
            res = sh.batch_render_dual(
                ind.chromo_g, ind.chromo_d, T_train, cfg.code_dim, cfg.target_dim
            )
            ind.F_raw = float(res["F"])
            c = np.asarray(res["c"])
            Fs, bd = shape_fitness(
                F_raw=ind.F_raw,
                c_matrix=c,
                classes=classes_train,
                len_g=len(ind.chromo_g),
                len_d=len(ind.chromo_d),
                cfg=fitness_cfg,
                seed_F_array=None,
            )
            ind.F_shaped = Fs
            ind.breakdown = bd
        except Exception as e:
            ind.F_raw = float("nan")
            ind.F_shaped = fitness_cfg.fallback_fitness
            ind.breakdown = {"error": str(e)[:200]}


# -------------------------------------------------------------- runner ---
def run(seed: int, mode: str, out_jsonl: Path, out_best: Path) -> dict:
    cfg = EaConfig.load(mode)
    fcfg = FitnessConfig.load()
    assets = MODE_ASSETS[mode]
    g_enc, g_dec = ensure_grammars(mode, cfg.code_dim, cfg.target_dim)

    T_path = assets["T_path"]
    if not T_path.is_file():
        raise SystemExit(
            f"{T_path} missing. For mode='pca' run scripts/pca_reduce.py; "
            f"for mode='raw' run scripts/embed_corpus.py."
        )
    T = np.load(T_path).astype(np.float64)
    classes = np.load(CLASSES_PATH)
    assert T.shape == (128, cfg.target_dim), (
        f"T shape mismatch for mode={mode}: got {T.shape}, expected "
        f"(128, {cfg.target_dim}). If you changed dims, regenerate the "
        f"grammars and the T matrix."
    )

    train_idx, test_idx = split_train_test(
        T.shape[0], cfg.train_fraction, seed=seed
    )
    T_train = np.ascontiguousarray(T[train_idx])
    classes_train = classes[train_idx]
    T_test = np.ascontiguousarray(T[test_idx])
    classes_test = classes[test_idx]
    print(
        f"[run_A1] seed={seed} split: train={len(train_idx)} test={len(test_idx)}"
    )

    from semiotic_hypercube import SemioticHypercube
    sh = SemioticHypercube(str(g_enc))
    sh.attach_decoder_grammar(str(g_dec))

    py_rng = random.Random(seed)

    # ---- Initial population --------------------------------------------
    pop: list[Individual] = []
    for _ in range(cfg.pop_size):
        g, d = fresh_pair(sh, py_rng, cfg.code_dim, cfg.target_dim)
        pop.append(Individual(chromo_g=g, chromo_d=d))
    evaluate_population(sh, pop, T_train, classes_train, cfg, fcfg)

    # ---- Evolution -----------------------------------------------------
    t0 = time.time()
    history = []
    with out_jsonl.open("w", encoding="utf-8") as log_f:
        for gen in range(cfg.n_generations + 1):
            pop.sort(key=lambda x: x.F_shaped, reverse=True)
            best = pop[0]
            F_best = best.F_shaped
            F_mean = float(np.mean([i.F_shaped for i in pop]))
            F_raw_best = best.F_raw
            len_g = len(best.chromo_g)
            len_d = len(best.chromo_d)
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seed": seed,
                "gen": gen,
                "F_best_shaped": F_best,
                "F_mean_shaped": F_mean,
                "F_best_raw": F_raw_best,
                "best_len_g": len_g,
                "best_len_d": len_d,
                "silhouette_cos": best.breakdown.get("silhouette_cos"),
            }
            history.append(entry)
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_f.flush()
            if gen % cfg.log_every == 0 or gen == cfg.n_generations:
                print(
                    f"[run_A1] seed={seed} gen={gen:3d} "
                    f"F_best_shaped={F_best:+.4f} (raw={F_raw_best:+.4f}) "
                    f"F_mean={F_mean:+.4f} "
                    f"best_len=({len_g},{len_d}) "
                    f"sil={entry['silhouette_cos']}"
                )

            if gen == cfg.n_generations:
                break

            # Build next gen: elitism + tournament-cross-mutate fill
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
                    # Validity gate: try a cheap dual-render on a tiny
                    # slice of T_train. Reject on failure (resample).
                    valid = False
                    for attempt in range(cfg.max_resample_attempts):
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
                        g, d = fresh_pair(sh, py_rng, cfg.code_dim, cfg.target_dim)
                        offspring.chromo_g = g
                        offspring.chromo_d = d
                    if len(next_pop) < cfg.pop_size:
                        next_pop.append(offspring)

            pop = next_pop
            evaluate_population(sh, pop, T_train, classes_train, cfg, fcfg)

    wall = time.time() - t0
    print(f"[run_A1] seed={seed} wall={wall:.1f}s")

    # ---- Test-split generalization ------------------------------------
    pop.sort(key=lambda x: x.F_shaped, reverse=True)
    best = pop[0]
    test_res = sh.batch_render_dual(
        best.chromo_g, best.chromo_d, T_test, cfg.code_dim, cfg.target_dim
    )
    F_test = float(test_res["F"])
    print(
        f"[run_A1] seed={seed} TEST F_raw={F_test:+.4f} "
        f"(train F_raw={best.F_raw:+.4f}  gap={best.F_raw - F_test:+.4f})"
    )

    out_best.write_text(
        json.dumps(
            {
                "seed": seed,
                "chromo_g": best.chromo_g,
                "chromo_d": best.chromo_d,
                "F_train_shaped": best.F_shaped,
                "F_train_raw": best.F_raw,
                "F_test_raw": F_test,
                "breakdown": best.breakdown,
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
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return {
        "seed": seed,
        "F_train_shaped": best.F_shaped,
        "F_train_raw": best.F_raw,
        "F_test_raw": F_test,
        "wall_s": wall,
    }


# --------------------------------------------------------------- main -----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["raw", "pca"], default="raw",
                    help="raw = A1 (T.npy, dim=1024); pca = A1.1 (T_pca.npy, dim=16)")
    args = ap.parse_args()

    tag = "" if args.mode == "raw" else f"_{args.mode}"
    out_jsonl = DEMO_DIR / f"runA1{tag}_seed{args.seed}.jsonl"
    out_best = DEMO_DIR / f"runA1{tag}_seed{args.seed}_best.json"

    summary = run(args.seed, args.mode, out_jsonl, out_best)
    print(f"[run_A1] summary: {summary}")
    print(f"[run_A1] wrote {out_jsonl}")
    print(f"[run_A1] wrote {out_best}")


if __name__ == "__main__":
    main()
