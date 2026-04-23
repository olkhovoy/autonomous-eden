# Alfred Tóth's Mathematical Semiotics — What Applies to Our Semiotic Hypercube Project

Source documents (local):
- `/home/user/mcs/theory/Strukturelle Realitaeten mit variabler Ordnung.pdf` (4 p., 10.02.2026)
- `/home/user/mcs/theory/Die kategorialen Transformationen der kybernetischen Semiotik.pdf` (87 p., book, 2020)

Author: Prof. Dr. Alfred Tóth (ученик Max Bense / Stuttgart School of semiotics).
Language: German. Tradition: Peirce → Bense → Walther → Tóth.

## 0. Attribution note [IMPORTANT]

Обложка книги 2020 г.: *"Transformations in a semiotic hypercube" (2009)*, © Semiotic
Technical Laboratory, Tucson, AZ. Термин **"semiotic hypercube" принадлежит
Тотy (2009)**, а не нам. В `SEMIOTIC-HYPERCUBE.md` и в README крейта/модуля
надо указать это как предшествующую работу (prior art), с которой мы
расходимся по цели (у нас вычислительная, у него классификационная
онтология). Это корректно научно и не требует изменений имён пакетов.

## 1. Формальная база Тотa

### 1.1. Пирсовская triадическая знак-релация
Z = (M, O, I): Mittel (носитель), Objekt (референт), Interpretant.
Каждая позиция имеет 3 трихотомии → 3×3×3 = **27 возможных троек**
(полная 27-я система), из которых 10 **"Peircesche Dualsysteme"**
реализованы в классической семиотике, а 17 — "mit Sternchen" (starred),
структурно возможны, но не интерпретируемы в Пирсе.

### 1.2. Пре-семиотическая цепь (Bense)
```
Ω  →  (O°  →  M°)  →  (M  →  O  →  I)
ontisch   presemiotisch        semiotisch
```
- **Ω** = абсолютный (апори́ческий) объект — нам недоступен
- **O°** = "вортетический" (disponibler) объект — субъективный объект
- **M°** = вортетические средства
- **M→O→I** = триадический знак
- Метаобъективация = map `g: O° → Z` (тетическая интродукция знака)
- Восприятие = map `h: Ω → O°` (нам недоступна "нейтрально")
- Полная цепь `f = g ∘ h` не биективна → отсюда рождаются
  "Gedankenzeichen" (знаки несуществующих объектов)

### 1.3. 9-операторная системная алгебра Σ*
```
Σ* = { S, ℛ[S,U], U,  ε, λ, σ, δ, ω, υ, ζ, ς, ρ }
```
где `S` — система, `U` — окружение, `ℛ[S,U]` — граница. Операторы:

| оп | смысл (нем.) | семантика | у нас |
|----|--------------|-----------|-------|
| ε  | Einbettung   | вложенность / системная иерархия | глубина SEQ-дерева, depth of nesting |
| λ  | Lage         | положение (λ_ex / λ_ad / λ_in = outside/boundary/inside) | **axis target type в грамматике** |
| σ  | Sortigkeit   | сорт/тип | класс op'а (AX vs MIX vs ROT vs ...) |
| δ  | Detachierbarkeit | отделимость | SUBTRACT / negate |
| ω  | Objektabhängigkeit | зависимость от других объектов | **CTRL(axis, code_idx) в A2 NC3** |
| υ  | Vermitteltheit | посредничество (через что проходит) | **MIX(i,j,α)** (медиируемая комбинация) |
| ζ  | Zugänglichkeit | доступность (readable/writable) | input vs output ports |
| ς  | Stufigkeit   | уровневая вертикальная адъюнкция | dim |
| ρ  | Reihigkeit   | горизонтальная адъюнкция | ADD / CONCAT / SEQ |

Это **готовая metaphysically-motivated taxonomy** для нашего op-set
грамматики. Наши ad-hoc операции (AX/ADD/SUB/MIX/ROT/FRAC) ложатся
в эту решётку почти 1-в-1.

### 1.4. Три типа пространственных отношений
- **Exessivität** (x ∊ ℛ[S,U]) — на границе/вне системы
- **Adessivität** (x ∩ ℛ[S,U] ≠ ∅) — пересекает границу
- **Inessivität** (x ∊ S) — внутри

Применительно к PCA-16: "inside" = top-16 SV-осей, "on-boundary" = оси
ближе к cutoff (0.65-0.70 explained variance), "outside" = осталось
30% вариативности.

### 1.5. Категориальная алгебра на {1,2,3}
Эндоморфизмы: `id1, id2, id3`
Морфизмы: `α: 1→2` (с инверсией `α°: 2→1`), `β: 2→3` (`β°: 3→2`),
композиции `βα: 1→3`, `α°β°: 3→1`.

Это **маленькая категория** с 3 объектами и 7 морфизмами. Она служит
генератором 27 эпикогнитивных сигнатур для 27 знаковых троек.

### 1.6. 270 Metaobjektivation-переходов
Полный каталог **всех возможных способов** превратить
воспринимаемый объект в знак в этой формальной системе:

```
27 Wahrnehmungsrelationen × 10 Erkenntnisrelationen = 270 transitions
```

Разбиты на 3 × 9 = 27 типов по мета-категориям:
- **Mediale / Objektale / Subjektale** (по позиции в triade)
- внутри каждой: qualitative, quantitative, essentielle, abstraktive,
  relative, komprehensive, konnexive, limitative, komplettierende
  (9 "аспектных" типов)

Это **готовый search space** для эволюционного поиска "способов
превращения эмбеддинга в код": вместо случайного поиска по дереву —
направленный перебор этих 270 шаблонов.

### 1.7. Eigenrealität ZZ (центральный математический объект)
В 3×3 матрице эпистемических функций
```
       Ω      Z      Σ
  Ω   ΩΩ    ΩZ    ΩΣ
  Z   ZΩ    ZZ    ZΣ
  Σ   ΣΩ    ΣZ    ΣΣ
```
диагональная клетка **ZZ** — "знак-в-самом-себе", посредник между
объективным субъектом (ΩΣ) и субъективным объектом (ΣΩ).

Это **буквально NC1 (Recursive Closure) из UMC**: `c ≈ G(D(c))`,
знак само-восстанавливается через петлю декодер→энкодер.
Тот же математический объект, разные имена.

### 1.8. Vermittlungslogik (Günther/Tóth)
Замена 2-значной L=[0,1] на 4-значный quadruple:
```
L1 = [0, [1]    L2 = [[1], 0]
L3 = [[0], 1]   L4 = [1, [0]]
```
Это capture "between" состояний, где объект и субъект asymmetrically
embedded. Теоретически применимо для NC2/NC3 (не-бинарные связи
между уровнями), но требует переосмысления fitness.

## 2. Что применимо к нашему проекту

### 2.1. [HIGH] Rename/reframe ops под Σ*-taxonomy (A2 scope)
В `rust/src/gggp/vector.rs` сейчас перечислены VectorOp'ы ad-hoc. При
переходе к A2 (добавляем CTRL/GATED_OP для NC3) — вместо придумывания
имён возьмём **9-операторную систему Тотa как рекомендованный
каталог**:
```
Einbettung (ε):     SEQ_DEPTH        -- бесплатно, уже есть
Lage (λ):           AX_EX, AX_AD, AX_IN   -- axis target: out/boundary/in
Sortigkeit (σ):     (tag in chromosome, не новая op)
Detachierbarkeit(δ):NEG, SUB
Objektabhaengig(ω): CTRL(axis, code_idx)   -- A2 new
Vermitteltheit (υ): MIX(i, j, alpha)        -- already
Zugaenglichkeit(ζ):READ_CODE(k), WRITE_OUT(k)
Stufigkeit (ς):     SCALE(factor)            -- already
Reihigkeit (ρ):     ADD, CONCAT              -- already
```

Преимущество: любой новый op должен укладываться в одну из 9 категорий,
что предотвращает рост op-set до хаотичного bloat (соответствует
правилу пользователя "одна функциональность — одна реализация").

### 2.2. [HIGH] UMC ↔ Tóth mapping как теоретическая рамка A2
| UMC constraint | Tóth concept | наша реализация |
|---|---|---|
| NC1 Recursive Closure | Eigenrealität ZZ | `F_nc1 = cos(c, G(D(c)))` |
| NC2 Unitary Integration | triadic M-O-I | (M=T, O=class, I=c); фитнес за consistency |
| NC3 Downward Causation | Subjektale Transition [αβ→...] | `CTRL(axis, code_idx)` op |
| NC4 Fixed-Point Stability | 10 Dualsysteme | уже есть в A1.1, F=0.92 |

Это **не косметика**: Tóth даёт формальные схемы всех допустимых
Subjektale Transitionen (subsection 4.3 в "kategoriale Transformationen"),
из которых можно читать **конкретные архитектуры** `CTRL`-операции.

### 2.3. [MEDIUM] 10 Dualsysteme как archetypes для кластеров
A1.1 дала ARI=1.0 на 8-классном k-means. Tóth предсказывает **10
"знаковых классов"** в полной Peircean комбинаторике. Эксперимент
для A3:
- corpus_v3 с 10 семантическими категориями (не 8)
- проверить, что emerge 10 кластеров (ARI вычислить vs 10 ground-truth)
- если yes — подтверждение, что emergent категоризация выбирает
  именно пирсовские "естественные сорта" знаков

### 2.4. [MEDIUM] Relationalzahlen как chromosome-level metadata
У Тотa относительные числа `1, 1-1, 1-2, 1-3, ...` описывают глубину
вложенности, а направления `←, 0, →` — тип отношения. Мы можем
добавить в каждый ген chromosome пару `(relnum, direction)`, что даст
структурную информацию для mutation: мутация "внутри" своего уровня
vs "мутация, пробивающая границу" — разные операторы по δ_mut.

### 2.5. [LOW] 270 Metaobjektivation каталог как directed search space
Вместо чистого EA-поиска по продукциям грамматики можно **гибридно**:
стартовая популяция = 270 канонических шаблонов Тотa → EA дорабатывает.
Цена: надо транслировать его символическую нотацию в наши VectorOp.
Это большая работа (≥1 неделя), оставляем на далёкое будущее (A4+).

### 2.6. [LOW] Vermittlungslogik для NC2
4-значная логика даёт формальный язык для "пограничных" состояний
между двумя классами эмбеддингов. В нашем корпусе каждый 16-й элемент
(paraphrase with ambiguous category) мог бы иметь logic-value L2
([1], 0) — "на границе класса 1". Это делает G4 (ARI) более нюансным.
Но усложняет fitness существенно. Откладываем до после A2.

## 3. Что НЕ применимо

- **17 starred sign classes** (*(3.2, 2.1, 1.1) и прочие). Это "formal
  possibilities, not realized in Peirce". Для нашего вычислительного
  проекта они не дают инженерного рычага — чисто исторический интерес.
- **Architectural metaphors** (дома, гаражи, ступеньки, парковки).
  Тотy нужны для иллюстрации абстрактных отношений. Нам не полезны.
- **Erotik der Helga Anders** (глава 7 книги). Пример, что
  нефункциональным отношениям тоже можно применить формализм. Никакой
  алгоритмической ценности.

## 4. Рекомендация в план

1. **Сейчас (пред-A2)**: обновить `SEMIOTIC-HYPERCUBE.md` — добавить
   блок "Prior Art: Tóth 2009" с корректной атрибуцией.
2. **В рамках A2/S1 (Rust op extension)**: до написания нового
   `CTRL`/`GATED_OP` — перечитать раздел 2.1 выше и привязать новые
   ops к таксономии Σ*, чтобы не расширять ad-hoc.
3. **В рамках A2/S4 (fitness)**: использовать mapping из раздела 2.2
   как комментарии в `fitness.py` — почему именно эти компоненты
   фитнеса соответствуют UMC constraints.
4. **В `docs/medp/ROOT.md`**: добавить ссылку на этот файл как
   background theory.
5. **A3+ как отдельная ветка**: "dualsysteme-10-classes" — эксперимент
   2.3 выше, требует corpus_v3 с 10 категориями.

## 5. Цитирование

```bibtex
@book{toth2020_transformations,
  author    = {Alfred T{\'o}th},
  title     = {Die kategorialen Transformationen der kybernetischen Semiotik},
  publisher = {Semiotic Technical Laboratory},
  address   = {Tucson, AZ},
  year      = {2020},
  note      = {Cover art ``Transformations in a semiotic hypercube'' (2009)},
}

@article{toth2026_strukturelle,
  author  = {Alfred T{\'o}th},
  title   = {Strukturelle Realit{\"a}ten mit variabler Ordnung},
  journal = {Electronic Journal for Mathematical Semiotics},
  year    = {2026},
  date    = {2026-02-10},
}
```

Precursor line: Peirce (1903) → Bense (1967, 1975) → Walther (1979) →
Günther (Polyvalent Logic, 1978) → Tóth (2009+).
