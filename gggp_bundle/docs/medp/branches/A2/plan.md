# A2 UMC Full Loop (NC1-4) on top of A1.1 PCA Space

Родительская ветка: `A1.1` (tag `gggp_bundle-medp/A1.1-success`).

## Мотивация

A1.1 операционализовала только **NC4** (Fixed-Point Stability):
`T_i ≈ D(G(T_i))` с F=0.92. Полный UMC требует добавить ещё три:

| neural constraint | смысл | что добавляет в PoC |
|---|---|---|
| **NC1** Recursive Closure | система самореферентна: код самовосстановим | `c_i ≈ G(D(c_i))` — дуальный fixed point |
| **NC2** Unitary Integration | композиция модальностей/контекстов | `D(α c_1 + β c_2)` ≈ смысл "смеси" `T_1 + T_2` |
| **NC3** Downward Causation | высший уровень влияет на низший | code c_i должен модулировать работу D (меняет операции) |
| **NC4** Fixed-Point Stability | базис | уже есть из A1.1, F=0.92 |

## Гипотеза A2

Одновременное удовлетворение NC1/NC2/NC3 поверх NC4 реализуемо в той же
axis-grammar на PCA-16, если:
  1. Добавить в fitness компоненты за NC1/NC2/NC3.
  2. Позволить decoder'у иметь input-ports, связанные с элементами `c`.
  3. Эволюционировать `(G, D)`-пару под многокритериальный fitness.

Прогноз: F_raw слегка просядет (с 0.92 до 0.70-0.85), но средняя
композиция `D(c_1 + c_2)` будет семантически ближе к `(T_1 + T_2)`,
чем случайное усреднение.

## Архитектура

### NC1 -- дуальная реконструкция
- Новый цикл: `c_i → D(c_i) = T_i' → G(T_i') = c_i'`.
- Fitness slot: `F_nc1 = mean_i cos(c_i, c_i')` ∈ [-1, 1].
- Мотивация: if `D` и `G` друг другу обратны, код устойчив к
  «внутреннему шуму» — операционный аналог self-identity.

### NC2 -- композиционность пары
- Для случайной пары `(i, j)` и весов `α = β = 0.5`:
  `c_mix = α c_i + β c_j`; `T_mix = D(c_mix)`.
- Ground-truth композиции нет, поэтому proxy: `T_mix` должен быть
  ближе к ПОЛНОМУ пулу `{T_i + T_j}` чем к случайному `T_k`.
  Метрика: triplet loss.
  `F_nc2 = mean triplet( D(0.5 c_i + 0.5 c_j), (T_i + T_j)/2, T_k_rand )`.

### NC3 -- downward influence
- Расширяем грамматику decoder'а: вводим op `CTRL(axis, src_code_idx)`,
  которая берёт конкретную компоненту кода `c[src_code_idx]` и использует
  её как множитель `δ` в axis-shift. Это делает D-программу *функционально
  зависимой* от содержимого c_i, а не просто от input vector.
- Уже сейчас `batch_render_dual` прокидывает `c_i` как input vector
  в D-дерево — это *частично* NC3 (D видит c_i как starting state).
- Усиление: добавить в D-грамматику `SCALE_BY_CODE(k)` / `ADD_CODE(axis, k)`.

### NC4 (carry-over из A1.1)
- Оставляем raw reconstruction: `F_nc4 = mean_i cos(D(G(T_i)), T_i)`.

## Fitness (multi-objective scalarized)

```
F_shaped = F_nc4 - w_len * len_penalty
         - w_class * silhouette_penalty
         + w_nc1 * F_nc1
         + w_nc2 * F_nc2
         + w_nc3 * F_nc3_signal
```

Где `F_nc3_signal` = доля D-операций, которые реально зависят от `c`
(структурная метрика, чтобы НЕ премировать «симулякры»).

Все веса `w_*` идут в `config/fitness.toml` с заявленными эволюционными
диапазонами для будущей мета-оптимизации (user rule).

## Декомпозиция задач (budget ≈ 6h)

| # | Задача | Est | Gate closed |
|---|---|---|---|
| S1 | Rust: добавить `CTRL`/`SCALE_BY_CODE`/`ADD_CODE` op в VectorOp + парсер | 1h | подготовка |
| S2 | `gen_neuro_grammar custom-nc3 16 <path>` режим + конфиг ops | 0.3h | подготовка |
| S3 | Python: extend `batch_render_dual` -> `batch_render_umc(chromo_g, chromo_d, T, code_dim, target_dim)` возвращает `{T_hat, c, c_hat_from_decode, F_nc1, F_nc4}` | 1h | подготовка |
| S4 | `fitness.py` -> `shape_fitness_umc(...)` с NC1/NC2/NC3 секциями | 0.7h | подготовка |
| S5 | `config/fitness.toml` -> расширить секцией `[nc_weights]` + ranges | 0.2h | подготовка |
| S6 | `run_A2.py` Runner (NSGA-II-лайт или взвешенная сумма — начнём с simple sum) | 1h | вход в G-eval |
| S7 | Run seed=0 -> сохраняем best.json + log.jsonl progress | 0.3h | — |
| S8 | eval_gates_A2 (G_NC1 > 0.5, G_NC2 > triplet-threshold, G_NC3 signal > 0.2, G_NC4 > 0.5) | 0.7h | формальные гейты |
| S9 | 5-seed replication + G5_A2 (stability) | 0.5h | G5 |
| — | Buffer + postmortem | 0.3h | — |

## Gates

| id | criterion | threshold | rationale |
|---|---|---|---|
| G_NC1 | `F_nc1 = mean cos(c, G(D(c)))` | > 0.50 | c должен быть «притяжим» через D-G loop |
| G_NC2 | triplet accuracy для композиций | > 0.55 | выше рандома (0.5) |
| G_NC3 | доля D-ops, зависящих от `c` | > 0.20 | не симулякр |
| G_NC4 | `F_nc4 = mean cos(T, D(G(T)))` | > 0.50 | сохраняет A1.1 качество (с запасом) |
| G_A2_stab | σ(F_overall) по 5 сидам | < 0.08 | чуть рыхлее A1.1 |
| G_A2_len | combined len | < 20 | больше чем A1.1 из-за NC3 ops |

## Риски

1. **NC3 cheating**: D может «притворяться» что использует `c`, но эффект
   операции на выход ничтожен. Митигация: контрольный запуск с
   randomized `c` — если F_nc4 не просел, значит D не реально зависит от c.
2. **NC2 тривиализация**: если `D(0.5 c_i + 0.5 c_j) ≈ (D(c_i) + D(c_j))/2`
   (D линеен), NC2 пройдёт бесплатно, но это не «настоящая» композиция.
   Митигация: ввести в грамматику `FRAC(f)`/`ROT` с нелинейным поведением.
3. **Многокритериальность**: простая взвешенная сумма может коллапсить
   в один критерий. Fallback: NSGA-II (pymoo), но это +2h бюджета.

## Preserve on Fail

- Если G_NC1 FAIL, но G_NC4 PASS — откат к A1.1 как operational baseline,
  fork A2.1 с явным G→D→G loss в обратном направлении.
- Если G_NC3 FAIL — признак, что grammar expressivity снова упирается
  в потолок; fork A2.2 с learned projection (уходит от axis-grammar).
- Если всё FAIL кроме NC4 — фиксируем как empirical negative result,
  возврат к F0 с уточнением теории (может, weighted sum — не тот
  способ компоновать UMC-критерии).

## Open question (требует решения перед S1)

Q1: используем ли мы **тот же T_pca.npy** (A1.1) или **ре-fit PCA на
большем корпусе** (256 или 512 paraphrases)? Больший корпус даст
лучший базис для NC2/NC3, но потребует новой `build_corpus_v2.py`.

Q2: **scalarization vs multi-objective**? Начать с weighted sum
(проще, быстрее) или сразу NSGA-II (pymoo, корректнее, +2h)?

Q3: **NC3 op set**: `CTRL(axis, code_idx)` достаточен или нужна ещё
`GATED_OP(op_id, code_idx)` (code выбирает какую операцию применить)?
GATED_OP гораздо богаче выразительно, но ломает статическую валидацию
дерева.
