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

### F1 = a′ — encoder = Ollama через provider-router (refinement 2026-04-22T16:45Z)

- Базовый провайдер: **Ollama** (`http://localhost:11434`), уже работает
  на рабочем RTX3090 пользователя; совпадает с тем, что уже использует
  `rust/src/bin/embedding_gggp.rs`.
- Архитектура: абстрактный `Provider` с двумя операциями (`embed`,
  `chat`) и pluggable backend'ами. Ollama — primary; OpenAI — stub для
  CI/сравнений; добавление новых — один класс. Конфиг:
  `gggp_bundle/config/providers.toml`.
- Embedding-модель для A1: **`ryanshillington/Qwen3-Embedding-0.6B`**,
  `dim = 1024`. Причины: 1024 dim сопоставим с MTEB-лидерами в классе,
  0.6B параметров быстрее на batch эмбеддингах, и главное — 4096 dim
  (8B-модель) гарантированно провалит G6 по механике (плоская алгебра
  из ≤12 ops не реконструирует точку в R^4096). 8B-модель (4096 dim)
  зарезервирована для A3/A4 с RGA-алгеброй.
- Chat-модель для γ.c генерации парафраз: **`Qwen3.6-35B-A3B:latest`**
  (MoE 35B total / 3B active), seed=42, temperature=0.
- Детерминизм: embedding запускается **один раз** на сессию, результат
  сохраняется как `T.npy` (128×1024) + `classes.npy`. Rust читает из
  файлов — никакого PyO3 callback'а из fitness loop. G5 (5 сидов)
  работает поверх зафиксированных эмбеддингов.

**Замена F1=a → F1=a′**: событие `threshold_adjusted` в `log.jsonl`
с обоснованием (pre-execution refinement, до первого gate_eval).

### F2 = a (γ.c LLMOnly) — corpus = paraphrase_gen (128 = 8 × 16)

- 8 seed-концептов, доменно-разнообразные:
  1. sort integers
  2. extract dates from text
  3. classify support tickets
  4. translate Russian to English
  5. detect anomalies in time series
  6. summarize an article
  7. route a payment
  8. parse a configuration file
- 16 парафраз per концепт через provider-router → Ollama chat
  (`Qwen3.6-35B-A3B:latest`, seed=42, temperature=0). Prompt-шаблон
  живёт в коде `scripts/build_corpus_v1.py` и коммитится вместе с
  корпусом (детерминизм воспроизводим на уровне «тот же prompt + тот же
  model snapshot = тот же output»).
- Ground-truth class = seed-id (0..7) → даёт G4 ARI бесплатно.
- Снапшот: `gggp_bundle/demos/semiotic_hypercube/corpus_v1.jsonl`
  коммитится в рамках T1. Downstream (T2..T10) reproducible поверх
  зафиксированного снапшота. Пересоздание корпуса = новый commit,
  новый `corpus_v2.jsonl`, не silent-regeneration.

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
| T0 | `scripts/providers.py` + `config/providers.toml` — provider-router (Ollama primary, OpenAI stub) | 0.5 | 0.5 | инфра для T1/T2 |
| T1 | `scripts/build_corpus_v1.py` → `corpus_v1.jsonl` (γ.c LLMOnly через Qwen3.6-35B-A3B, seed=42) | 0.75 | 1.25 | вход G1 |
| T2 | `scripts/embed_corpus.py` — Ollama embed (Qwen3-Embedding-0.6B) → `T.npy` (128×1024) + `classes.npy` | 0.25 | 1.5 | вход G1/G2 |
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
| G1 | 17:00 | `|M|=128` ∧ `dim(T_i)=1024` | 100% |
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

**T7**: собрать `scripts/run_A1.py` — 80/20 split, EA-цикл
(selection + crossover + mutation над парой (G, D)), fitness через
`SemioticHypercube.batch_render_dual` + `scripts/fitness.shape_fitness`.
Коммит-префикс обычный (`feat(A1): ...`).

### Artifact paths locked in T6

- `gggp_bundle/demos/semiotic_hypercube/grammar_encoder.cfg`
  (dim=16, gitignored; регенерируется
  `cargo run --release --bin gen_neuro_grammar -- encoder <path>`).
- `gggp_bundle/demos/semiotic_hypercube/grammar_decoder.cfg`
  (dim=1024, gitignored).
- `gggp_bundle/config/fitness.toml` — веса fitness-shaper'а с
  задекларированными evolutionary ranges (alpha_len, L_max,
  beta_class, gamma_seed).

### Backtrack triggers for A1

Эти события автоматически запускают `medp(A1): backtrack`:

| событие | действие |
|---|---|
| G3 fail (F <= F_0 + 0.10 после полного бюджета T7) | fork → A1.1 с CMA-ES hook |
| G6 fail при G3 pass (длина > 12) | fork → A1.2 с усиленным alpha_len |
| grammar coverage < 3 уникальных op типов на топ-5 индивидах | fork → A1.3 с пересмотренной грамматикой |
