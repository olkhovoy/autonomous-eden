# 01. Evaluate Gate — оценка чекпоинта

Шаблон для фальсифицируемой оценки одного гейта. Вызывается моделью,
скриптом или человеком. Результат — строго numeric + pass/fail +
запись в журнал.

---

## Вход

- `branch_id` — активная ветка.
- `gate_id` — один из G1..G6 (§7 ROOT.md) или расширенный gate,
  определённый в `branches/<id>/plan.md`.
- `artifacts_path` — абсолютные пути к артефактам, собранным веткой
  (corpus.jsonl, best_pair.json, runs logs и т.п.).
- `deadline_exceeded` — bool. Если true и метрика не собрана → fail.

## Алгоритм оценки

1. Прочитать из `docs/medp/ROOT.md` §7 спецификацию `gate_id`:
   `metric, threshold, comparator, when_after_start`.
2. Собрать метрику из `artifacts_path` ровно по определению §7.
   Запрещено менять формулу метрики «на лету». Если формула
   недостаточно специфицирована — fail с `reason="metric_underspecified"`
   и предложение уточнения в `on_fail_next` → `threshold_adjusted` или
   `meta_backtrack`.
3. Применить compar: `value comparator threshold`.
4. Записать результат:
   - Append в `log.jsonl`:
     ```
     {"ts":"<ISO>","event":"gate_eval","branch":"<id>","author":"<who>","gate":"<gate_id>","result":"<pass|fail>","value":<x>,"threshold":<y>,"comparator":"<cmp>"}
     ```
   - Обновить `branches/<id>/checkpoints.md`:
     строкой `- [<x|—>] <gate_id> value=<v> threshold=<t> at <ts>`.
   - Коммит (точный формат):
     `medp: gate <gate_id> <result> value=<v> threshold=<t> branch=<id>`.

## Специальные случаи

- **G2 (baseline record)** — не fail-гейт. Записать `result:"record"`
  с `value=F_0`, порог=null, comparator=null.
- **`value` нестабильно между прогонами** — записать среднее и std;
  при `std > 0.5 * |threshold - value|` считать fail
  (нестабильность не даёт принять решение).
- **Артефакт отсутствует** — fail с `reason="artifact_missing: <path>"`.

## Рамка ответа

```
## Gate <gate_id> evaluation

- Метрика (из ROOT §7): <определение>
- Собранное значение: <v> (std=<σ> над N=<k> прогонами, если применимо)
- Порог: <t>
- Компаратор: <cmp>
- Результат: <pass|fail|record>
- Причина (при fail): <одна фраза>

## Записи (готовы к применению)

log.jsonl append:
<JSON-строка>

checkpoints.md append:
<markdown-строка>

commit message:
<строка>

## Следующее действие по ROOT §8

<ссылка на строку таблицы переходов или "продолжить до следующего гейта">
```

Ожидать подтверждения пользователя перед append в `log.jsonl` и коммитом.
Запрет на автоматическое выполнение backtrack в этом шаблоне — это
делается в `02_backtrack.md`.
