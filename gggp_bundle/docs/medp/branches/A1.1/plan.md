# A1.1 PCA-Space SCL PoC (refinement of A1 after G3 FAIL)

Родительская ветка: A1 (tag `gggp_bundle-medp/A1-partial`).

Гипотеза (из A1 diagnostic):
  Потолок F=0.17 в A1 — следствие того, что грамматика оперирует индексами
  координат, а 80%+ энергии эмбеддинга лежит в верхних сингулярных
  направлениях. Переход в PCA-базис даёт грамматике шанс работать в
  «правильных» осях.

Предсказание:
  Если гипотеза верна — G3 PASS в PCA-пространстве.
  Если G3 по-прежнему FAIL — грамматическая алгебра (даже с MIX/ROT/FRAC)
  принципиально не способна восстанавливать, нужна A1.2 (learned
  projection via input bank) или A1.4 (смена парадигмы).

## Locked Decisions

### F1.1 — PCA preprocessing
- Fit sklearn PCA(n_components=16) on T (128 × 1024).
- Produce T_pca.npy shape (128, 16), float32, сохраняется вместе с
  explained_variance_ratio_.json (для аудита).
- Центрирование PCA по умолчанию (sklearn); `mean_T_pca` будет ~0 по
  построению → F_0 переопределяется как `mean_i ||T_pca_i||_norm=1`
  после L2-норм. мы нормализуем T_pca построчно, чтобы метрика F была
  сопоставима с A1.
- Не коммитим T_pca.npy в git (gitignored, регенерируем из T.npy).

### F2.1 — Reconstruction dims
- target_dim = 16 (в PCA-пространстве)
- code_dim = 8 (2:1 компрессия; это заставляет G научиться сжимать)
- Оба grammar'а регенерируем через `gen_neuro_grammar`:
    encoder: dim=8
    decoder: dim=16

### F3.1 — Fitness baseline recompute
- F_0_pca = mean_i cos(mean_T_pca_normalized, T_pca_normalized_i),
  аналогично G2 в A1. Это новый anchor для G3.
- G3 threshold остаётся формулой `F > F_0 + 0.10`, но сам F_0 меняется.

## Декомпозиция задач (dev budget 2.5h)

| # | Задача | Est | Закрывает |
|---|--------|-----|-----------|
| R1 | `scripts/pca_reduce.py` → `T_pca.npy` (128×16) + explained_variance.json | 0.25 | вход G1.1 |
| R2 | Re-gen grammars dim=8 и dim=16 через CLI | 0.1 | вход R3 |
| R3 | Обновить `run_A1.py` с флагом `--mode pca` (читает T_pca.npy, code_dim=8, target_dim=16) | 0.5 | вход R4/R5 |
| R4 | G1.1 eval (structural: n=128, dim=16) + G2.1 recompute F_0_pca | 0.25 | anchor G3.1 |
| R5 | Re-run EA seed=0 в PCA-mode | 0.5 | вход G3.1 |
| R6 | G3.1 / G4.1 / G6.1 eval через `eval_gates_g3_g4_g6.py` (параметризовать) | 0.5 | — |
| R7 | Multi-seed (5 сидов) + G5.1 eval | 0.25 | G5 |
| — | Buffer: postmortem, решение promote/backtrack | 0.15 | — |

## Gate Schedule (A1.1)

| id | criterion | threshold |
|----|-----------|-----------|
| G1.1 | n=128 AND dim_pca=16 | 100% |
| G2.1 | F_0_pca recorded | (record) |
| G3.1 | F > F_0_pca + 0.10 | > F_0_pca + 0.10 |
| G4.1 | k-means(k=8) ARI на c_i | > 0.30 |
| G5.1 | σ(F) по 5 сидам | < 0.05 |
| G6.1 | combined_len(best) | < 12 |

## Предварительная проверка достижимости

- Spectral top-16 в A1 diagnostic = 0.803 (80% энергии).
- В 16-dim PCA space L2-норма полностью захватывает эту энергию.
- Для идентичной функции G=D=identity: F = 1.0 в PCA-16.
- Для случайной G,D: F_random ≈ 0 (как в A1).
- Для G4 ARI: в A1 достигнут 0.80 без PCA; в PCA-space ожидаем не хуже.
- Для G6: A1 достиг len=4; PCA просто уменьшает размерность, на длину
  программы не влияет.

## Preserve on Fail

Если G3.1 снова FAIL (margin > -0.1 в PCA-space):
  `log.jsonl` entry: `gate_eval G3.1 fail` →
  `fork_proposed F1.2_mix_bank` (переход к A1.2) ИЛИ
  `backtrack_to_parent F0` с entry `architectural_dead_end_axis_grammar`.

Если G3.1 PASS, но с margin < +0.05:
  `threshold_adjusted` с обоснованием и переход к A1.2 или промоутим как успех.

## Следующее действие

~~R1~~ написать `scripts/pca_reduce.py` и породить `T_pca.npy`. **[DONE]**

## Status update (final)

Ветка закрыта как **SUCCESS (6/6 gates)**. Все R1-R7 завершены.
См. `postmortem.md` для детальных чисел и рекомендаций по следующим
веткам (A2 / A1.2 / B1).
