# Plan A1 — GGGP-decoder × synthetic × flat algebra

Копия Plan-узла §6.2 ROOT.md, расширенная зафиксированными решениями
pre-A1 scouting (2026-04-22) и декомпозицией задач в рамках бюджета.

## Метаданные

- `branch_id`: `A1`
- `parent`: `root`
- `git realization`: см. §6.6/F0.d ROOT.md — commit-prefix `medp(A1):` +
  теги `gggp_bundle-medp/A1-*` на `main`.
- `budget_dev_hours`: **8**
- `budget_compute_hours`: **4**
- `start_ts`: 2026-04-22T16:30:00Z (зафиксирован в `log.jsonl`)
- `status`: `active`

## Зафиксированные решения (pre-A1 scouting)

Все три закрыты явным подтверждением пользователя. Изменение любой из
этих решений после start = `threshold_adjusted`-эквивалент (требует
отдельной записи в `log.jsonl` с обоснованием) или backtrack в A2/A3/A6.

### F1 = a — encoder = sentence-transformers

- Модель: `sentence-transformers/all-MiniLM-L6-v2`.
- Размерность: `dim = 384`.
- Детерминировано (torch seed фиксирован, eval-mode, CPU).
- Без Ollama-инфраструктуры.
- Bridge: embedding считается в Python один раз per session и
  сохраняется как `T.npy` (128 × 384). Rust-код читает из файла —
  это избавляет от PyO3 callback overhead на каждом fitness eval
  и делает G5 честным (одинаковый encoder между сидами).

### F2 = a — corpus = paraphrase_gen (128 = 8 × 16)

- 8 seed-концептов, доменно-разнообразные:
  1. sort integers
  2. extract dates from text
  3. classify support tickets
  4. translate Russian to English
  5. detect anomalies in time series
  6. summarize an article
  7. route a payment
  8. parse a configuration file
- 16 парафраз per концепт: active/passive, formal/casual, synonym subst.
- Ground-truth class = seed-id (0..7) → даёт G4 ARI бесплатно.
- Артефакт: `gggp_bundle/demos/semiotic_hypercube/corpus_v1.jsonl`
  (commit в рамках A1, не генерируется на каждом запуске).

### F3 = a — decoder = shared-genome co-evolution

- `GpIndividual.trees[0] = G` (encoder-дерево).
- `GpIndividual.trees[1] = D` (decoder-дерево).
- Одна и та же vector-грамматика (AX/SCALE/NORM/MIX/ROT/FRAC/ZERO).
- Кроссовер и мутация применяются к обоим деревьям — co-evolution.
- Fitness = `mean_i cos(T_i, D(G(T_i)))
             − λ_1 · mean_i len(G(T_i))
             − λ_2 · H({c_i})
             − λ_3 · std_{seeds} F`.
  `λ_1=0.01, λ_2=0.05, λ_3=0.5` — начальные значения, могут быть
  перенастроены CMA-ES'ом на `continuous_weights` (см. §5 HYBRID).

## Декомпозиция задач (dev budget 8 h)

| # | Задача | Est | Cum | Закрывает |
|---|--------|-----|-----|-----------|
| T1 | Python-скрипт: `scripts/build_corpus_v1.py` → `corpus_v1.jsonl` (128 × {text, class}) | 1.0 | 1.0 | вход G1 |
| T2 | Python-скрипт: `scripts/embed_corpus.py` — sentence-transformers → `T.npy` (128×384) + `classes.npy` | 0.5 | 1.5 | вход G1/G2 |
| T3 | **G1 eval**: размер и dim проверка, запись в `checkpoints.md` + `log.jsonl` | 0.25 | 1.75 | G1 |
| T4 | **G2 eval**: baseline `F_0 = mean_i cos(mean_T, T_i)`, запись | 0.25 | 2.0 | G2 |
| T5 | Rust: `GpIndividual::render_dual(dim) → (V_enc, V_dec)` и обновление fitness-коллбэка в `python_api.rs` | 1.5 | 3.5 | G3 |
| T6 | Rust/Python: fitness с penalty-ами + hook для CMA-ES на `continuous_weights` | 0.5 | 4.0 | G3 |
| T7 | Python-runner `scripts/run_A1.py` — 80/20 split, эволюция, метрика `F` на train-split | 1.0 | 5.0 | G3 |
| T8 | **G3 eval** + Python: k-means(k=8) на `{c_i}` train-split, ARI vs classes, запись | 1.0 | 6.0 | G4 |
| T9 | **G4 eval** + 5 сидов (0,1,2,3,4) параллельно, σ(F) | 1.0 | 7.0 | G5 |
| T10 | **G5 eval** + гистограмма длин `len(G(T_i))` + avg_len | 0.5 | 7.5 | G6 |
| — | Buffer: **G6 eval**, postmortem/promote-commit, tag, сводка в `checkpoints.md` | 0.5 | 8.0 | — |

Compute: ≤ 12k fitness evals × 5 сидов ≈ 3.5 h CPU (есть запас 0.5 h).

## Расписание гейтов

Из §7 ROOT.md, дедлайны — от `start_ts = 2026-04-22T16:30:00Z`:

| id | deadline (UTC) | metric | threshold |
|----|----------------|--------|-----------|
| G1 | 17:00 | `|M|=128` ∧ `dim(T_i)=384` | 100% |
| G2 | 18:30 | `F_0` record | — |
| G3 | 20:30 | `F_train` > `F_0 + 0.10` | gt |
| G4 | 2026-04-23 00:30 | `ARI({c_i}, y)` > 0.30 | gt |
| G5 | 2026-04-23 08:30 wall | `σ(F)` < 0.05 (5 сидов) | lt |
| G6 | 2026-04-23 16:30 wall | `avg_len(c_i)` < 12 | lt |

При пересечении wall-deadline без закрытия гейта — fail (§3).

## preserve_on_fail

Артефакты, которые **нельзя** удалять при backtrack'е (§3 Archive):

- `gggp_bundle/demos/semiotic_hypercube/corpus_v1.jsonl` — reusable в A2/A3/A6/A7.
- `gggp_bundle/demos/semiotic_hypercube/T.npy` + `classes.npy`.
- `branches/A1/artifacts/baseline_F0.json`.
- `branches/A1/artifacts/best_pair.json` (chromosomes G/D + fitness).
- `branches/A1/artifacts/fitness_curve.svg`.
- `branches/A1/checkpoints.md`.

## on_fail_next (из §8 ROOT.md)

- G1 fail → **A1** повтор (один раз, `budget_extended`).
- G3 fail → **A3** (RGA).
- G4 fail → **A2** (линейный декодер).
- G5 fail → **A2**.
- G6 fail → **A3**.

## Следующее действие

**T1**: создать `gggp_bundle/scripts/build_corpus_v1.py`, запустить,
закоммитить результат. Коммит-префикс обычный (не `medp(A1):`) —
`medp(A1):` зарезервирован только для событий протокола (start /
gate / backtrack / promote / budget_extended).
