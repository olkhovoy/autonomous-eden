# MEDP — Meta-Evolutionary Development Protocol

Корневой документ протокола. Любая новая сессия должна быть способна
восстановить полное состояние разработки, прочитав этот файл + `log.jsonl` +
содержимое активных `branches/<branch_id>/`.

Ссылки на связанные документы:

- `../SEMIOTIC-HYPERCUBE.md` — исходный концепт Semiotic Hypercube.
- `../UMC_HYPOTHESES.md` — гипотезы UMC и критерии NC1/NC2/NC3/NC4.
- `../EMBEDDING_GGGP.md` — текущая реализация embedding-фенотипа.
- `../EVOLUTIONARY_ENGINE_ROADMAP.md` — продуктовый фокус движка.
- `../RUST_PORT_STATUS.md` — индекс статуса Rust-порта.
- `research/toth-semiotics.md` — prior art: Alfred Tóth, Peirce/Bense
  семиотическая традиция и её применимость к нашим грамматикам.

---

## 1. Корневая интенция (root intent)

Доказать реализуемость **Semiotic Closure Loop (SCL)** как минимального
убедительного PoC концепта Semiotic Hypercube.

Операционное утверждение, которое PoC должен подтвердить или опровергнуть:

> GGGP способен породить **язык** — компактное символическое
> представление, выживающее round-trip'у `decode(encode(m)) ≈ m`
> на фиксированном корпусе смыслов `M`, при штрафах за длину,
> энтропию и межсидовую дисперсию.

Неудача PoC — валидный результат, если причина зафиксирована и
позволяет переформулировать гипотезу. Успех PoC — валидный результат,
только если все гейты пройдены без ретроактивного ослабления порогов.

---

## 2. Принцип

MEDP = применение тех же принципов (UMC NC1/NC4, SCL round-trip) к самому
процессу разработки:

- NC1: план порождает артефакты → артефакты порождают метрики → метрики
  правят план. Петля замкнута в репо, а не в chat-state.
- NC4: идентичное состояние репо → идентичное следующее действие
  (детерминированное правило `on_fail_next`).
- SCL round-trip: `decode(plan) = факты в git + log.jsonl`;
  `encode(факты) = обновлённый plan`. Расхождение — сигнал к meta-backtrack.

Следствие: chat-context между сессиями намеренно **не переносится**.
Переносится только репо + `log.jsonl`.

---

## 3. Словарь (определяется здесь один раз)

- **Plan** — узел дерева разработки. Обязательные поля:
  `intent, parent, branch_id, budget_{dev_hours, compute_hours},
  checkpoints[], preserve_on_fail[], on_fail_next[], units[]`.
- **Unit** — атомарная единица работы внутри Plan; структура и правила
  декомпозиции/роутинга агентов см. в правиле
  `.cursor/rules/PLAN-DECOMPOSITION-AND-AGENT-ROUTING.mdc`. Обязательные
  поля: `id, title, depends_on, parallel_group, recommended_agent,
  rationale_for_agent, est_minutes, artifacts_{in,out}, preserve_on_fail`.
  `budget_dev_hours` плана = сумма `est_minutes` по **critical path**,
  а не по всем unit'ам.
- **Fork** — точка ветвления с ≥2 технически разными альтернативами,
  имеющими сопоставимые ожидаемые utilities.
- **Gate (Checkpoint)** — фальсифицируемое численное условие с дедлайном:
  `{id, when_after_start, metric, threshold, comparator, cost_to_eval}`.
- **Backtrack** — переход на ближайшего `on_fail_next` при провале гейта.
  Обязательны: `postmortem.md`, тег `medp/archive/<branch_id>-fail-<gate_id>`,
  соответствующая запись в `log.jsonl`.
- **Archive** — неизменяемый тег + ветка `medp/archive/*` на провалившемся
  состоянии. **Запрещено удалять**. Это корпус для будущей MEDP-v2 и
  входные данные для `learnings-researcher`-подобных процессов.
- **Promote** — все гейты активной ветки пройдены → `git merge` в родителя
  + тег `medp/promote/<branch_id>`.
- **Budget** — верхняя граница `{dev_hours, compute_hours}`. Истечение
  бюджета = автоматический fail последнего несобранного гейта.

Правило: **ни один `Plan` не живёт дольше своего `budget`**. Продление
бюджета требует явного коммита `medp: budget_extended <branch> reason=<...>`
с обоснованием — и сразу засчитывается как частичный fail в журнал.

---

## 4. Структура директории

```
docs/medp/
  ROOT.md                 # этот файл
  log.jsonl               # append-only журнал событий (источник истины)
  prompts/
    00_kickoff.md         # восстановление контекста новой сессией
    01_evaluate_gate.md   # оценка гейта
    02_backtrack.md       # постмортем и выбор on_fail_next
    03_fork.md            # предложение нового форка
  branches/
    <branch_id>/
      plan.md             # копия Plan-узла + ссылки на артефакты
      checkpoints.md      # состояние всех гейтов ветки
      postmortem.md       # создаётся только при fail
```

Git:

- Активная ветка: `medp/<branch_id>`.
- Архив провала: тег `medp/archive/<branch_id>-fail-<gate_id>` на HEAD провалившейся ветки.
- Успешный promote: merge в родителя + тег `medp/promote/<branch_id>`.
- Корневой тег: `medp/root` на commit’е bootstrap (`medp: bootstrap protocol`).

Формат commit-сообщений (обязательный для всех MEDP-событий):

```
medp: <event> <branch> [<details>]

Примеры:
  medp: bootstrap protocol
  medp: start A1
  medp: gate G3 pass value=0.42 threshold=0.10 branch=A1
  medp: gate G4 fail value=0.18 threshold=0.30 branch=A1
  medp: backtrack A1 reason=G4-fail next=A3
  medp: promote A1
  medp: budget_extended A1 reason="CMA-ES converged slower than estimated"
```

`git log --grep='^medp:'` становится живым журналом разработки.

---

## 5. Формат `log.jsonl`

Append-only. Одна JSON-запись на строку. Источник истины по принятым
решениям. При расхождении с `plan.md` / `postmortem.md` побеждает `log.jsonl`.

Общие поля:

- `ts` — ISO-8601 UTC.
- `event` — из списка ниже.
- `branch` — `branch_id` или `"root"`.
- `author` — `"human"` или `"model:<slug>"`.

События и обязательные специфичные поля:

| event | обязательные поля |
|---|---|
| `medp_bootstrap` | `note` |
| `branch_start` | `parent`, `budget_dev_h`, `budget_compute_h` |
| `gate_eval` | `gate`, `result` (`pass`/`fail`), `value`, `threshold`, `comparator` |
| `backtrack` | `reason`, `next` |
| `promote` | `merged_into` |
| `threshold_adjusted` | `gate`, `old`, `new`, `reason` |
| `budget_extended` | `old_dev_h`, `new_dev_h`, `reason` |
| `fork_proposed` | `new_fork_id`, `alternatives`, `rationale` |
| `meta_backtrack` | `level`, `reason` |

Запрет: редактировать существующие строки. Только append. Исправление
ошибочной записи — новая запись `correction` со ссылкой на `corrects_ts`.

---

## 6. Дерево форков

### 6.1 Root

- `branch_id`: `root`
- `parent`: `—`
- `intent`: Фаза A корневой интенции (§1) — Closure Loop PoC.
- `budget_dev_hours`: 40
- `budget_compute_hours`: 16
- `children (первая волна)`: `A1`, `A2`, `A3`
- `reserve`: `A4..A12` (см. §6.5)
- `on_fail_next`: `meta_backtrack` (см. §10)

### 6.2 A1 — GGGP-decoder × synthetic × flat algebra

- `F1=a`: декодер — второй GGGP-tree в той же vector-грамматике.
- `F2=a`: корпус — синтетический, 128 парафраз из `demo_targets.txt`.
- `F3=a`: алгебра — текущая плоская (`AX/SCALE/NORM/MIX/ROT/ZERO`).
- `budget_dev_hours`: 8
- `budget_compute_hours`: 4
- Обоснование приоритета: минимальное отклонение от реализованного
  `embedding_gggp`, нет новых зависимостей, все гейты оцениваются оффлайн.
- `preserve_on_fail`: `corpus.jsonl`, `baseline_F0.json`, `best_pair.json`,
  кривая фитнеса в `svg`.

### 6.3 A2 — Linear-decoder × synthetic × flat

- `F1=b`: декодер — `W ∈ R^{d×d}`, оптимизируется CMA-ES на `continuous_weights`.
- `F2=a`: синтетический корпус (тот же).
- `F3=a`: плоская алгебра.
- `budget_dev_hours`: 12
- `budget_compute_hours`: 6
- Обоснование: null-hypothesis для симметричного GGGP-декодера.
  Если линейного слоя достаточно — вся «грамматика-декодер» избыточна.
- `preserve_on_fail`: оптимизированная матрица `W`, распределение
  компонент `c_i` по осям.

### 6.4 A3 — GGGP-decoder × synthetic × Recursive Glyph Algebra

- `F1=a`: симметричный GGGP-декодер.
- `F2=a`: синтетический корпус.
- `F3=b`: RGA — `CROSS/CIRCLE/TRIANGLE/SELF` с bounded amplitude sum.
- `budget_dev_hours`: 20
- `budget_compute_hours`: 6
- Обоснование: прямая проверка гипотезы «фрактальная алгебра даёт больший
  $F$ при меньшей длине программы».
- `preserve_on_fail`: определения операторов в `vector.rs`,
  доказательство сходимости (таблица amplitude-sum по глубине),
  mandala-визуализация `c_i`.

### 6.5 Reserve (A4..A12)

Не работа-план, а **гарантия возврата**. Активируется только если
первая волна исчерпана без promote.

| id | F1 | F2 | F3 | триггер активации |
|---|---|---|---|---|
| A4 | b | a | b | A3 fail G3 → линейный декодер + RGA |
| A5 | c | a | a | A1+A2 оба fail G4 → LLM-decoder на синтетике |
| A6 | a | b | a | A1 fail G4 при стабильном G5 → внешний корпус может дать структуру |
| A7 | b | b | a | A2 fail G4 → внешний корпус |
| A8 | a | b | b | A3 fail G4 → внешний корпус |
| A9 | b | b | b | A4 fail G4 |
| A10 | c | a | b | A5 pass + нужно проверить фрактальность |
| A11 | c | b | a | A5 pass + нужна обобщаемость |
| A12 | c | b | b | последний резерв; максимальная стоимость |

### 6.6 Мини-форк F0 (разрешён до bootstrap)

- **F0.a** — MEDP живёт в `gggp_bundle/docs/medp/`. **Принято.**
- Основание: ближе к коду SCL; мигрирует вместе с bundle при
  `SDK_REPO_SPLIT_PLAN.md`; минимальное связывание.
- F0.b, F0.c — зафиксированы как отвергнутые альтернативы для аудита.
- **F0.d** — MEDP-ветки реализованы как **commit-prefix + tag** на `main`,
  а не как реальные git-ветки, пока `gggp_bundle/` живёт внутри
  родительского репозитория `/home/user/mcs`. **Принято.**
  Формат: `medp(<branch>): <event> ...` для коммитов,
  `gggp_bundle-medp/<branch>-<tag>` для тегов (пример:
  `gggp_bundle-medp/root`, `gggp_bundle-medp/A1-start`,
  `gggp_bundle-medp/A1-fail-G4`).
  Переход на реальные ветки `medp/<branch>` — автоматически после
  исполнения `SDK_REPO_SPLIT_PLAN.md`.

---

## 7. Гейты первой волны

Единая сетка для A1/A2/A3; значения порогов зафиксированы **до старта**
ветки и могут быть изменены только через явное событие
`threshold_adjusted` в `log.jsonl` с обоснованием.

Обозначения: `t` = время старта ветки; `T_i = E(m_i)` — embedding строки
из корпуса; `c_i = G(T_i)` — код; `T̂_i = D(c_i)` — реконструкция; `F` —
средний по корпусу `cos(T_i, T̂_i)` с штрафами (см. `docs/SEMIOTIC-HYPERCUBE.md`
§«Семиотическое замыкание» и `EMBEDDING_GGGP.md` §7).

| id | дедлайн | метрика | порог | компаратор | что значит fail |
|---|---|---|---|---|---|
| G1 | `t + 30 min` | `\|M\| = 128` и `dim(T_i) == dim_model` для всех `i` | `100%` match | `eq` | баг в пайплайне корпуса |
| G2 | `t + 2 h` | baseline `F_0 = mean_i cos(mean_T, T_i)` | `—` | `record` | не-fail; фиксация нуля для сравнения |
| G3 | `t + 4 h` | mean `F(G,D)` на train-split (80%) | `F > F_0 + 0.10` | `gt` | пара не побеждает тривиальный baseline |
| G4 | `t + 8 h` | ARI кластеризации `{c_i}` vs ground-truth классы корпуса | `ARI > 0.30` | `gt` | язык не отражает структуру данных |
| G5 | `t + 16 h` | std `F` по 5 независимым сидам | `σ < 0.05` | `lt` | результат — шум, не язык |
| G6 | `t + 24 h` | средняя длина программы в Op | `len < 12` | `lt` | «язык» не сжимает; не выполнен критерий квалиа-компрессии |

Оценка каждого гейта выполняется по `prompts/01_evaluate_gate.md` и
протоколируется записью `gate_eval` в `log.jsonl` + коммитом
`medp: gate <id> <result> value=<x> threshold=<y> branch=<branch_id>`.

---

## 8. Правила переходов (on_fail_next)

Детерминированные. При идентичном состоянии журнала выбирается
одинаковая следующая ветка:

| из ветки | провал гейта | следующий `branch_id` |
|---|---|---|
| A1 | G1 | A1 (повтор с фиксом; budget_extended один раз) |
| A1 | G2 | — (G2 не fail-гейт) |
| A1 | G3 | A3 (меняем алгебру: плоских ops не хватило) |
| A1 | G4 | A2 (меняем декодер: симметричный GGGP слишком выразителен) |
| A1 | G5 | A2 (линейный слой обычно стабильнее) |
| A1 | G6 | A3 (фрактальная алгебра даёт краткость) |
| A2 | G3 | A3 |
| A2 | G4 | A6 (внешний корпус) |
| A2 | G5 | — (CMA-ES стабилен по определению; при fail — meta_backtrack) |
| A2 | G6 | A3 |
| A3 | G3 | A4 |
| A3 | G4 | A8 (внешний корпус) |
| A3 | G5 | A4 (линейный декодер + RGA) |
| A3 | G6 | — (RGA по построению сжимает; при fail — meta_backtrack) |
| любая | `budget_exceeded` | `on_fail_next` её последнего незакрытого гейта |

Конфликты таблицы разрешаются в пользу меньшего `branch_id` резервного ряда
(A4 раньше A5 и т.д.).

---

## 9. Протокол на git + markdown (без MCP)

См. §6.6 / F0.d: пока bundle живёт внутри родительского репо,
«ветка» = commit-prefix `medp(<id>):` + теги `gggp_bundle-medp/<id>-*`
на `main`. Инструкции ниже применяются с этой подстановкой.

1. **Старт ветки.** Создать `branches/<id>/plan.md` по шаблону из §6.2.
   Append `{event:"branch_start",...}` в `log.jsonl`. Коммит:
   `medp(<id>): start`. Тег: `git tag gggp_bundle-medp/<id>-start`.
2. **Работа до гейта.** Обычные коммиты по сути задачи (не MEDP-префикс).
3. **Оценка гейта.** Открыть `prompts/01_evaluate_gate.md`, подставить
   контекст, выполнить (руками/моделью/скриптом). Результат →
   `branches/<id>/checkpoints.md` + `log.jsonl`. Коммит:
   `medp: gate <gate_id> <result> value=<x> threshold=<y> branch=<id>`.
4. **Pass.** Продолжать до следующего гейта. После последнего гейта —
   §6-promote.
5. **Fail.** Открыть `prompts/02_backtrack.md`. Сгенерировать
   `branches/<id>/postmortem.md`. Коммит:
   `medp(<id>): backtrack reason=<gate_id>-fail next=<next_id>`.
   Тег: `git tag gggp_bundle-medp/<id>-fail-<gate_id>`.
   Старт следующей ветки по §9.1.
6. **Promote.** Merge-коммит в `main` не требуется (работа уже на main).
   Тег: `git tag gggp_bundle-medp/<id>-promote`. Append
   `{event:"promote",...}`.

Запрет на rebase веток `medp/*` и на force-push в `medp/archive/*`.
`medp/archive/*` — неизменяемый слой истории.

---

## 10. Meta-backtrack

После исчерпания `A1..A12` без единого promote **не** открывается `A13`.
Вместо этого:

- `log.jsonl` получает запись `{event:"meta_backtrack", level:1, reason:"all_A_branches_failed"}`.
- `ROOT.md` пересматривается целиком: под вопрос ставится само разбиение
  пространства решений и/или формулировка корневой интенции в §1.
- Результат — новая версия `ROOT.md` с тегом `medp/root-v2` + запись
  `meta_bootstrap` в журнале.

Граница: MEDP не гарантирует, что первая редакция `ROOT.md` корректно
нарезала пространство. Она гарантирует только, что провал будет явным и
восстановимым.

---

## 11. Самопроверка соответствия собственным критериям

| Критерий | Как проверен в MEDP |
|---|---|
| Фальсифицируемость | Все гейты численные с порогами и дедлайнами (§7) |
| Компрессия | MEDP = этот файл (≤400 строк) + 4 шаблона + `log.jsonl` |
| Round-trip | Новая сессия восстанавливает состояние только из файлов (см. `prompts/00_kickoff.md`) |
| NC4-стабильность | Таблица переходов (§8) детерминирована по состоянию `log.jsonl` |
| Одна реализация | Формат Plan — один; формат Gate — один; журнал — один |
| Конфигурируемость | Все пороги/бюджеты/переходы — в §6–§8, а не в коде |

---

## 12. Следующее действие (live pointer)

При любом прочтении этого файла — проверить `log.jsonl`, найти последнюю
запись и действовать согласно таблице переходов (§8) или правилам фаз (§9).

На момент bootstrap: активная ветка — `root`, все A-ветки — pending.
Следующее действие — §9.1 со стартом `A1`.
