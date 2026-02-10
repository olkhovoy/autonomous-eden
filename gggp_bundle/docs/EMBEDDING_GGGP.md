# Embedding GGGP: техническая документация

Документ описывает `embedding_gggp.rs` — режим GGGP, где фенотипом является **вектор**
(embedding), а fitness — **cosine similarity** к целевому embedding от Ollama.

Исходник: `core/rust/src/bin/embedding_gggp.rs`

---

## 1) Назначение

`embedding_gggp` — это демонстрация **GGGP → векторного фенотипа**, где грамматика
генерирует последовательность операций над вектором, а эволюция подбирает программу,
максимизирующую сходство с целевым embedding целевого текста.

Цели:
- проверить применимость GGGP к «семиотическим» представлениям;
- поддержать PoC Semiotic Hypercube;
- создать основу для дальнейших расширений (VSG, FRAC‑грамматики).

---

## 2) Архитектура (вкратце)

1. **Целевой embedding** берётся из Ollama (`/api/embeddings`).
2. **Грамматика** определяет последовательность операций над вектором.
3. **GGGP** эволюционирует дерево, выводимое в текст (program string).
4. **Vector interpreter** применяет программу к нулевому вектору.
5. **Fitness** = cosine similarity(candidate, target).

---

## 3) Запуск

Минимальный запуск:

```
cargo run --bin embedding_gggp -- --target "sort a list of integers"
```

Дополнительные параметры:

```
--model     <name>   Ollama embedding model (default: all-minilm:33m)
--url       <url>    Ollama embeddings endpoint (default: http://localhost:11434/api/embeddings)
--seed      <u64>    RNG seed for GGGP
--gens      <n>      Generations (default: 200)
--pop       <n>      Population size (default: 120)
--elite     <n>      Elite size (default: 8)
--max-ops   <n>      Grammar depth cap (default: 24)
--axis-step <f64>    Axis step for {AXIS_STEP} placeholder (default: 1.0)
--value-step <f64>   Value step for {VALUE_STEP} placeholder (default: 0.1)
--crossover <f64>    Crossover rate (default: 0.7)
--mutation <f64>     Mutation rate (default: 0.3)
--target    <text>   Target text for embedding (required)

--cfg        <path>  Load grammar and/or run config from .cfg (binary)
--dump-cfg   <path>  Save resolved grammar to .cfg
--save-best  <path>  Save best vectors as JSONL on improvements
--plot-2d    <path>  Save 2D projection (SVG or CSV)
--plot-3d    <path>  Save 3D projection (CSV)
```

---

## 3.1 Run config via .cfg

If `--cfg` is provided, the file is scanned for a node named `EmbeddingGGGP`
(case-insensitive). Attributes found there can override defaults when the
corresponding CLI flag is not set. The file may also contain a `Grammar` node,
which will be used as the grammar source.

Supported attributes (with or without leading `@`):

- `Target` (Str)
- `Model` (Str)
- `Url` (Str)
- `Seed` (Int)
- `Gens` (Int)
- `Pop` (Int)
- `Elite` (Int)
- `MaxOps` (Int)
- `AxisStep` (Real)
- `ValueStep` (Real)
- `Crossover` (Real)
- `Mutation` (Real)
- `SaveBest` (Str)
- `Plot2D` (Str)
- `Plot3D` (Str)

If no `EmbeddingGGGP` node exists, the root node is checked for these attributes.

---

## 4) Формат выхода

### 4.1 Логи

В консоли печатаются:
- target, dim, seed
- best fitness каждые 10 поколений
- итоговая best fitness и best program

### 4.2 JSONL лучших решений (`--save-best`)

На каждое улучшение записывается строка JSONL:

```
{
  "generation": 42,
  "fitness": 0.61234,
  "dim": 384,
  "vector": [ ... ],
  "program": "...",
  "target_text": "...",
  "model": "all-minilm:33m",
  "seed": 42
}
```

### 4.3 2D plot (`--plot-2d`)

- Если путь оканчивается на `.svg`: сохраняется SVG с линией траектории.
- Имена SVG нумеруются по итерациям улучшений:
  `best_2d.00123.svg`
- Если путь не `.svg`: создаётся CSV `generation,fitness,x,y`.

### 4.4 3D plot (`--plot-3d`)

CSV: `generation,fitness,x,y,z`.

---

## 5) Грамматика

По умолчанию грамматика строится программно (`build_vector_grammar`).
Можно загрузить внешнюю через `--cfg`.

### 5.1 Основные правила (по умолчанию)

- `START -> SEQ`
- `SEQ -> OP SEQ | OP`
- `OP` включает операции:
  - `AX axis val`
  - `SCALE val`
  - `NORM`
  - `MIX axis axis weight`
  - `ROT axis axis angle`
  - `FRAC exponent`

### 5.2 Placeholder‑ы в тексте

При загрузке .cfg поддерживаются подстановки:

- `{DIM}`        → `dimension - 1`
- `{AXIS_STEP}`  → значение `--axis-step`
- `{VALUE_STEP}` → значение `--value-step`

Подстановки выполняются **до** `parse_text` и `calc_lengths`.

---

## 6) Интерпретатор операций

Начальное состояние: `vector = [0, 0, ...]`

### 6.1 AX (Add axis)

```
AX <axis> <val>
```
Добавляет значение `val` в координату `axis`.

### 6.2 SCALE

```
SCALE <val>
```
Умножает все координаты на `val`.

### 6.3 NORM

Нормирует вектор до unit‑norm (если норма > 1e‑12).

### 6.4 MIX

```
MIX <axis_a> <axis_b> <weight>
```
Смешивает две координаты: `weight` в [0..1].

### 6.5 ROT

```
ROT <axis_a> <axis_b> <angle_degrees>
```
Поворот в плоскости (axis_a, axis_b).

### 6.6 FRAC

```
FRAC <exponent>
```
Нелинейное масштабирование каждой координаты: `sign(v) * |v|^exponent`.

---

## 7) Fitness

```
fitness = cosine_similarity(candidate, target)
```

Cosine similarity в диапазоне [-1, 1]. В текущем PoC оптимизация направлена
на максимум (обычно положительные значения при разумной эволюции).

---

## 8) Ограничения и текущие компромиссы

- GGGP использует `ThreadRng` без внешнего seed — полная детерминированность
  между запусками пока не гарантирована.
- Выбор осей дискретный (`axis_step`) — это компромисс для устойчивой эволюции.
- Векторные программы могут быть разреженными — норма улучшения растёт с бюджетом.

---

## 9) Пример .cfg (шаблон)

Можно сгенерировать:

```
--dump-cfg /tmp/vector_grammar.cfg
```

И затем редактировать в текстовом виде через `cfgdump`.

---

## 10) Рекомендации для «вау‑демо»

- Попробовать 3 разных цели (см. `core/demos/semiotic_hypercube/demo_targets.txt`).
- Увеличить `gens` и `pop` для более плотного вектора.
- Сохранять SVG на каждый новый best — это наглядный прогресс.

---

## 11) Связанные файлы

- `core/demos/semiotic_hypercube/README.txt`
- `core/demos/semiotic_hypercube/run_demo.sh`
- `core/demos/semiotic_hypercube/demo_targets.txt`
