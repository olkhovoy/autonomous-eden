# Техспек Rust-порта: CFG + GGGP

Документ описывает реализованную функциональность и технические детали Rust-порта:
бинарный формат .cfg, GGGP-движок и CLI-утилиты.

---

## 1) Что уже реализовано (Rust)

1. **Бинарный формат .cfg (деревья узлов)**  
   Реализованы чтение/запись v1/v2 формата, поддержка всех типов данных и массивов, индексация
   атрибутов/детей по имени, корректный round-trip.

2. **CLI `cfgdump`**  
   Конвертация .cfg в текстовые форматы: `txt`, `md`, `html`. Работает со stdin/stdout.

3. **GGGP engine (порт uGGGP)**  
   Парсер грамматики, построение дерева из хромосомы, генерация текста, поддержка числовых
   диапазонов, "group tags", генетические операторы (crossover, mutation).

4. **CLI `gggp`**  
   Генерация текста, кроссовер, мутация. Поддержка RNG seed, выбор grammar node, подключение
   глобального cfg (PluginHost).

5. **GGGP glyph demo**  
   Прототип визуальной демонстрации: эволюция глифов по целевым формам (Glagolitic-inspired),
   вывод SVG.

6. **Тесты**  
   - `core/rust/tests/roundtrip_sample.rs` — round-trip для реального .cfg.  
   - `core/rust/tests/synthetic_tree.rs` — синтетический набор типов и массивов.

---

## 2) Формат .cfg (бинарный)

Файл содержит дерево узлов (Node). Каждый узел имеет:

- `name` (bytes, регистронезависимо для индекса),
- `attrs` (список атрибутов),
- `children` (список дочерних узлов),
- `sorted` (флаг отсортированности детей),
- `duplicates` (сколько одинаковых имён допустимо),
- `global_id`, `modified` (в v2).

См. реализацию: `core/rust/src/storage/codec.rs`, `core/rust/src/storage/node.rs`.

### 2.1 Версия v2 (актуальная)

**Header**: строка  
`Custom file v.0002. Contact author for specification. E-mail: <olkhovoy@gmail.com>`

**Общий принцип**: имена передаются через таблицу ID (NameWriter/NameReader).  
Имя пишется единожды, далее ссылкой по ID.

**Node v2 layout**:

1. `name_id` (i32, -1 = конец списка)  
2. `global_id` (i32)
3. **attrs**: для каждого атрибута  
   - `attr_name_id` (i32)  
   - `type` (u8)  
   - `value` (в зависимости от типа)
4. `attr_name_id = -1` (конец атрибутов)
5. `sorted` (u8: 0/1)
6. `duplicates` (i32)
7. `modified` (i32)
8. **children**: для каждого ребёнка  
   - `child_name_id` (i32)  
   - рекурсивно Node v2
9. `child_name_id = -1` (конец детей)

Имена нормализуются в lower-case для индекса (см. `normalize_name`).

### 2.2 Версия v1 (legacy)

**Header**: строка  
`Custom file format. Contact author for specification. E-mail: <olkhovoy@gmail.com>`

**Структура**:

1. Таблица имён:  
   - `count` (i32)  
   - для каждого имени: `len` (i32) + `bytes`
2. **Node v1 layout** (имя узла задаётся индексом при чтении потомком):
   - `attr_count` (i32)
   - `data_len` (i32)
   - `data` (bytes)
   - далее `attr_count` атрибутов из data-блока:
     - `name_index` (u16)
     - `type` (u8)
     - `value` (по типу)
   - `child_count` (i32)
   - `sorted` (u8)
   - `duplicates` (i32)
   - для каждого ребёнка: `name_index` (i32) + Node v1 рекурсивно

### 2.3 Типы данных (u8)

```
0  Unknown
1  Int (i32, LE)
2  Real (f64, LE)
3  Bool (u8: 0/1)
4  Str (len u32 + bytes)
5  Time (f64, LE)  // Delphi datetime
6  Sing (f32, LE)
7  Bin (len u32 + bytes)
8  IntArray (len u32 + i32 * len)
9  RealArray (len u32 + f64 * len)
10 BoolArray (len u32 + raw bytes)
11 StrArray (count u32 + repeated: len u32 + bytes)
12 TimeArray (len u32 + f64 * len)
13 SingArray (len u32 + f32 * len)
14 BinArray (count u32 + record_size u32 + raw bytes)
```

### 2.4 Время (Delphi)

`Time` хранится как `f64` в днях с эпохи `1899-12-30 00:00:00`,  
строковое представление (для вывода): `YYYY-MM-DD HH:MM:SS`.

---

## 3) CLI `cfgdump`

Исходник: `core/rust/src/bin/cfgdump.rs`

```
cfgdump [-f txt|md|html] [-o OUTPUT] <input.cfg|->
```

- Выводит дерево узлов с атрибутами и типами.
- `-` на входе читает stdin.
- `-o` пишет в файл, иначе stdout.

---

## 4) GGGP: грамматика, хромосома, генерация

Исходник: `core/rust/src/gggp/mod.rs`

### 4.1 Узел Grammar в .cfg

Типовой вид:

```
Grammar
  @MaxDepth (Int)
  @MaxCrossoverNodes (Int)
  @MaxMutationNodes (Int)
  @OPTIMIZE (Bool)
  @Chromosome (Str)
  RULES
    START
      CHOICES
        0 -> Text = "<SEQ>"
    SEQ
      CHOICES
        0 -> Text = "<SEG> <SEQ>"
        1 -> Text = "<SEG>"
    ...
```

Ключевые поля:

- `MaxDepth`, `MaxCrossoverNodes`, `MaxMutationNodes` — ограничения дерева.
- `OPTIMIZE` и `Chromosome` — логика выбора хромосомы (см. CLI `gggp`).
- `RULES` → символы → `CHOICES` → варианты (имя дочернего узла = число).

### 4.2 Формат `Text` и ссылки `<...>`

`Text` содержит ссылки на другие символы:

```
<SYMBOL>
<len from=0.2 to=1 inc=0.2>
```

Парсер (`parse_text`) строит `REFS` и сохраняет ссылки в виде `GpRef`.  
Доступны параметры:

- `from`, `to`, `inc` — числовой диапазон (создаёт виртуальный символ).
- `optimize=true|false` или `opt=true|false` — управляющая форма.
- `maxdepth` — локальное ограничение глубины.

Экранирование `<` и `>` в тексте: `<<` и `>>` (после генерации заменяются обратно).

### 4.3 Хромосома

Строка вида `3-160-191-1-61-...`

- Парсится как последовательность `i32`.
- Каждый ген выбирает **номер варианта** в `CHOICES` (имя узла = число).
- Развёртывание идёт при построении дерева в глубину.

### 4.4 Генерация текста

- Из хромосомы строится `GpTree`.
- `indented_text()` формирует текст по `Text` у choice.
- `%Name%` заменяется именем метода (путь к grammar node).
- `%Id%` заменяется на последовательные id (0,1,2...).
- Строки нормализуются на CRLF (`\r\n`) и всегда заканчиваются CRLF.
- `%group`/`%ind` заменяются через `gp_data_from_config` (опционально, через global cfg).

---

## 5) CLI `gggp`

Исходник: `core/rust/src/bin/gggp.rs`

```
gggp [text] [-c <chromosome>] [-g <grammar_path>] [--global-cfg <path>] [-o OUTPUT] <input.cfg|->
gggp crossover -c <chromosome> -d <chromosome> [-g <grammar_path>] [--global-cfg <path>] [-o OUTPUT]
gggp mutate [-c <chromosome>] [-g <grammar_path>] [--global-cfg <path>] [-o OUTPUT]

Опции:
  -c, --chromosome   хромосома, если OPTIMIZE=true или если нужно override
  -d, --chromosome2  для crossover
  -g, --grammar      путь к узлу Grammar (если их несколько)
  --global-cfg       глобальный cfg для group/index resolution
  -s, --seed         RNG seed для reproducibility
```

Логика выбора хромосомы:

- Если `OPTIMIZE=true`, хромосому нужно передать через `-c`.
- Если `OPTIMIZE=false`, используется `@Chromosome` в узле Grammar.

---

## 6) Генетические операторы

В `gggp` реализованы:

- `crossover_individuals` — обмен поддеревьями между двумя особями.
- `mutate_individual` — замена случайного поддерева новым.

Оба оператора работают на `GpIndividual` и используют RNG (seed в CLI).

---

## 7) GGGP glyph demo (глаголические глифы)

Исходник: `core/rust/src/bin/glyph_demo.rs`

Идея: эволюция глифов как визуальных "квалиа-петель".

- **Грамматика:** turtle-команды `F len`, `L ang`, `R ang`, `Z` (close).
- **Цели:** набор глаголических архетипов (AZ, BUKI, VEDI, GLAGOL, DOBRO, EST, IZHE, SLOVO, ON).
- **Fitness:**  
  - сходство bitmap (F1),
  - замкнутость (closure),
  - сложность (penalty),
  - симметрия,
  - различимость между глифами (distinctness).

Выводится SVG: серым — target, чёрным — лучший результат.

---

## 8) Входные данные для GGGP

Обязательные входы:

- `input.cfg` с узлом `Grammar` и `RULES`.
- Хромосома (CLI или `@Chromosome` в узле Grammar).

Опциональные:

- `--global-cfg` для замены `%group`/`%ind`.
- `-g` если в cfg несколько Grammar узлов.

---

## 9) Fitness: как считаем

В текущих прототипах:

1. **Glyph demo**  
   `fitness = 0.45 * sim + 0.3 * closure + 0.15 * complexity + 0.1 * symmetry`  
   + дополнительный бонус за различимость глифов.

2. **Основной GGGP (торговые стратегии / код)**  
   Зависит от задачи. Здесь нужен отдельный фитнес-драйвер (performance, стабильность, риск).

---

## 10) Примечания

- Документ описывает текущее состояние кода.  
- Все пути и источники указаны в коде и документах проекта.
