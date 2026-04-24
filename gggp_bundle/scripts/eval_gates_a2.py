"""
gggp_bundle/scripts/eval_gates_a2.py

MEDP A2 / S3b -- formal gate evaluation for branch A2.

Gates (from docs/medp/branches/A2/plan.md §Gates):
  G_NC1      F_nc1      > 0.50     code is D-G attractable
  G_NC2      F_nc2      > 0.55     triplet acc above chance (0.5)
  G_NC3      F_nc3      > 0.20     >=20% of D-ops depend on c
  G_NC4      F_nc4      > 0.50     A1.1 quality preserved (with margin)
  G_A2_stab  sigma(F_umc) < 0.08   reasonable seed-stability
  G_A2_len   max(len_g + len_d) < 20   combined chromosome size bound

Plus a diagnostic for Risk 1 (NC3 cheating): for each seed,
recompute NC4 after shuffling c row-wise. If shuffled-NC4 stays
within <epsilon_delta> of original NC4, D did not actually depend
on c (the NC3 signal was symbolic only). Emits a warning flag per
seed, not a gate -- the gates are purely score-based.

Inputs:
  demos/semiotic_hypercube/runA2_seed{0,1,2}_best.json
  demos/semiotic_hypercube/T_v2_pca.npy
  demos/semiotic_hypercube/grammar_{encoder,decoder}_a2*.cfg

Outputs:
  demos/semiotic_hypercube/a2_gate_report.json
  demos/semiotic_hypercube/a2_gate_report.md

Run:
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/eval_gates_a2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
SEEDS = (0, 1, 2)

# Gate thresholds (locked per A2 plan.md §Gates).
THRESHOLDS = {
    "G_NC1":      ("F_nc1",   "> 0.50"),
    "G_NC2":      ("F_nc2",   "> 0.55"),
    "G_NC3":      ("F_nc3",   "> 0.20"),
    "G_NC4":      ("F_nc4",   "> 0.50"),
    "G_A2_stab":  ("sigma_F", "< 0.08"),
    "G_A2_len":   ("max_len", "< 20"),
}

# Risk-1 diagnostic: minimum NC4 drop after c-shuffle for D to count as
# genuinely using c. 0.10 is conservative but not extreme.
RISK1_DELTA_MIN = 0.10


def load_sh():
    from semiotic_hypercube import SemioticHypercube
    sh = SemioticHypercube(str(DEMO_DIR / "grammar_encoder_a2.cfg"))
    sh.attach_decoder_grammar(str(DEMO_DIR / "grammar_decoder_a2_nc3.cfg"))
    return sh


def _cos_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    dots = np.einsum("ij,ij->i", a, b)
    out = np.zeros_like(dots, dtype=np.float64)
    safe = denom > 1e-12
    out[safe] = dots[safe] / denom[safe]
    return out


def nc4_with_shuffled_c(sh, best: dict, T: np.ndarray) -> float:
    """Risk-1 control: re-decode each row using c from a permuted index.

    Procedure:
      1. Run batch_render_dual to get the baseline c matrix.
      2. Shuffle c rows by a fixed permutation (seed-derived).
      3. For each i, call render_tree_with_input(chromo_d, target_dim,
         c_shuffled[i], role="decoder") to get T_hat_shuffled[i].
      4. Return mean cos(T[i], T_hat_shuffled[i]).

    If D actually uses c, T_hat_shuffled should look random-ish and the
    mean cosine should be near 0 (or at worst, inter-class cosine,
    which was -0.11 for our PCA corpus). If D does NOT use c (operates
    only on the implicit seeding of the state via c[:code_dim]), the
    explicit ADDC/SBC/... ops will make this identical to the original
    NC4 and the drop will be zero.

    NB: the IMPLICIT state seeding (state[:8] = input = c) IS using c.
    Any nonzero NC3 op on top adds additional c-dependence. The
    diagnostic detects the second source of c-dependence, which is
    what G_NC3 claims the decoder exhibits.
    """
    code_dim = int(best["ea_config"]["code_dim"])
    target_dim = int(best["ea_config"]["target_dim"])
    res = sh.batch_render_dual(
        best["chromo_g"], best["chromo_d"], T, code_dim, target_dim
    )
    c = np.ascontiguousarray(np.asarray(res["c"], dtype=np.float64))
    n = c.shape[0]
    rng = np.random.default_rng(best["seed"])
    perm = rng.permutation(n)
    # Guard against a fixed-point permutation (rare for N>=8 but possible).
    for retry in range(10):
        if not np.any(perm == np.arange(n)):
            break
        perm = rng.permutation(n)
    c_shuf = np.ascontiguousarray(c[perm])

    T_hat = np.empty((n, target_dim), dtype=np.float64)
    for i in range(n):
        T_hat[i] = sh.render_tree_with_input(
            best["chromo_d"], target_dim, c_shuf[i], role="decoder"
        )
    return float(_cos_rowwise(T, T_hat).mean())


def check_gate(rule: str, value: float) -> bool:
    """Evaluate 'rule' (e.g. '> 0.50', '< 0.08') on value.

    A tiny hand-rolled evaluator beats eval() / ast.literal_eval for
    this simple case; the two supported operators are exactly what
    plan.md §Gates uses.
    """
    op, threshold = rule.split()
    thr = float(threshold)
    if op == ">":
        return value > thr
    if op == "<":
        return value < thr
    if op == ">=":
        return value >= thr
    if op == "<=":
        return value <= thr
    raise ValueError(
        f"check_gate: unsupported operator '{op}' in rule '{rule}'. "
        f"Extend THRESHOLDS with a matching branch if needed."
    )


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))
    T = np.load(DEMO_DIR / "T_v2_pca.npy").astype(np.float64)

    sh = load_sh()
    per_seed_rows = []
    for seed in SEEDS:
        p = DEMO_DIR / f"runA2_seed{seed}_best.json"
        if not p.is_file():
            print(f"[S3b] WARN missing {p.name}, skipping seed {seed}")
            continue
        best = json.loads(p.read_text())
        tr = best["train"]
        te = best["test"]
        nc4_shuf = nc4_with_shuffled_c(sh, best, T)
        nc4_orig = float(tr["F_nc4"])
        per_seed_rows.append({
            "seed": seed,
            "F_umc": float(tr["F_umc"]),
            "F_shaped": float(tr["F_shaped"]),
            "F_nc1": float(tr["F_nc1"]),
            "F_nc2": float(tr["F_nc2"]),
            "F_nc3": float(tr["F_nc3"]),
            "F_nc4": nc4_orig,
            "F_test_shaped": float(te["F_shaped"]),
            "train_test_gap": float(tr["F_shaped"]) - float(te["F_shaped"]),
            "len_g": len(best["chromo_g"]),
            "len_d": len(best["chromo_d"]),
            "combined_len": len(best["chromo_g"]) + len(best["chromo_d"]),
            "nc4_shuffled_c": nc4_shuf,
            "nc4_drop": nc4_orig - nc4_shuf,
        })
        print(
            f"[S3b] seed={seed} "
            f"NC1={tr['F_nc1']:+.3f} NC2={tr['F_nc2']:+.3f} "
            f"NC3={tr['F_nc3']:.3f} NC4={nc4_orig:+.3f} "
            f"NC4_shuf={nc4_shuf:+.3f} "
            f"drop={nc4_orig - nc4_shuf:+.4f}"
        )

    if not per_seed_rows:
        raise SystemExit(
            "[S3b] no runA2_seed*_best.json files found. "
            "Run run_A2.py first."
        )

    # Aggregate across seeds for gates.
    mean_scores = {
        "F_nc1":  float(np.mean([r["F_nc1"]  for r in per_seed_rows])),
        "F_nc2":  float(np.mean([r["F_nc2"]  for r in per_seed_rows])),
        "F_nc3":  float(np.mean([r["F_nc3"]  for r in per_seed_rows])),
        "F_nc4":  float(np.mean([r["F_nc4"]  for r in per_seed_rows])),
    }
    sigma_F = float(np.std([r["F_umc"] for r in per_seed_rows]))
    max_len = int(max(r["combined_len"] for r in per_seed_rows))
    gate_values = {
        **mean_scores,
        "sigma_F": sigma_F,
        "max_len": float(max_len),
    }

    gate_results = []
    all_pass = True
    for gate_id, (metric, rule) in THRESHOLDS.items():
        val = gate_values[metric]
        passed = check_gate(rule, val)
        all_pass = all_pass and passed
        gate_results.append({
            "gate": gate_id,
            "metric": metric,
            "rule": rule,
            "value": val,
            "pass": bool(passed),
        })
        print(
            f"[S3b] {gate_id:<12} {metric:<8} = {val:+.4f}  "
            f"{rule:<7}  -> {'PASS' if passed else 'FAIL'}"
        )

    # Risk-1 diagnostic across seeds.
    risk1_per_seed = []
    for r in per_seed_rows:
        genuinely_uses_c = r["nc4_drop"] >= RISK1_DELTA_MIN
        risk1_per_seed.append({
            "seed": r["seed"],
            "nc4_drop": r["nc4_drop"],
            "genuinely_uses_c": bool(genuinely_uses_c),
        })
    genuine_count = sum(1 for x in risk1_per_seed if x["genuinely_uses_c"])
    print(
        f"[S3b] Risk-1 NC3-cheating diagnostic: "
        f"{genuine_count}/{len(risk1_per_seed)} seeds show NC4 drop "
        f">= {RISK1_DELTA_MIN} under c-shuffle "
        f"(decoder genuinely uses c beyond the implicit state seed)."
    )

    verdict = "PASS" if all_pass else "FAIL"
    print(f"[S3b] A2 branch verdict: {verdict}")

    report = {
        "n_seeds": len(per_seed_rows),
        "per_seed": per_seed_rows,
        "aggregate": {
            "mean_F_nc1": mean_scores["F_nc1"],
            "mean_F_nc2": mean_scores["F_nc2"],
            "mean_F_nc3": mean_scores["F_nc3"],
            "mean_F_nc4": mean_scores["F_nc4"],
            "sigma_F_umc": sigma_F,
            "max_combined_len": max_len,
        },
        "thresholds": {
            gid: {"metric": m, "rule": r} for gid, (m, r) in THRESHOLDS.items()
        },
        "gates": gate_results,
        "risk1": {
            "delta_min": RISK1_DELTA_MIN,
            "per_seed": risk1_per_seed,
            "n_genuine": genuine_count,
        },
        "verdict": verdict,
    }
    out_json = DEMO_DIR / "a2_gate_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[S3b] wrote {out_json}")

    # Markdown.
    md = [
        "# A2 — Gate evaluation",
        "",
        f"**Verdict: {verdict}** ({sum(g['pass'] for g in gate_results)}/"
        f"{len(gate_results)} gates pass)",
        "",
        "## Gates",
        "",
        "| gate | metric | rule | value | verdict |",
        "|---|---|---|---|---|",
    ]
    for g in gate_results:
        md.append(
            f"| {g['gate']} | {g['metric']} | `{g['rule']}` | "
            f"{g['value']:+.4f} | "
            f"{'**PASS**' if g['pass'] else '**FAIL**'} |"
        )
    md += [
        "",
        "## Per-seed summary",
        "",
        "| seed | F_umc | F_shaped | NC1 | NC2 | NC3 | NC4 | "
        "F_test | gap | lenG | lenD | NC4_shuf | drop |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in per_seed_rows:
        md.append(
            f"| {r['seed']} | {r['F_umc']:+.4f} | "
            f"{r['F_shaped']:+.4f} | "
            f"{r['F_nc1']:+.3f} | {r['F_nc2']:+.3f} | "
            f"{r['F_nc3']:.3f} | {r['F_nc4']:+.3f} | "
            f"{r['F_test_shaped']:+.4f} | "
            f"{r['train_test_gap']:+.4f} | "
            f"{r['len_g']} | {r['len_d']} | "
            f"{r['nc4_shuffled_c']:+.3f} | {r['nc4_drop']:+.4f} |"
        )
    md += [
        "",
        "## Risk 1 — NC3 cheating diagnostic",
        "",
        "Procedure: shuffle the c matrix row-wise, re-decode each row",
        f"with the decoder, recompute NC4. If the drop is < "
        f"{RISK1_DELTA_MIN:.2f}, the decoder is NOT genuinely using c",
        "beyond the implicit state-seeding phase.",
        "",
        "| seed | NC4_drop | genuine c-use? |",
        "|---|---|---|",
    ]
    for r in risk1_per_seed:
        md.append(
            f"| {r['seed']} | {r['nc4_drop']:+.4f} | "
            f"{'YES' if r['genuinely_uses_c'] else 'NO'} |"
        )
    md += [
        "",
        f"**{genuine_count}/{len(risk1_per_seed)}** seeds show genuine "
        "c-use (drop ≥ "
        f"{RISK1_DELTA_MIN:.2f}).",
        "",
    ]
    out_md = DEMO_DIR / "a2_gate_report.md"
    out_md.write_text("\n".join(md))
    print(f"[S3b] wrote {out_md}")

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
