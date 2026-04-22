# 02. Backtrack — постмортем и выбор следующей ветки

Вызывается после `gate_eval` с `result:"fail"` или при
`budget_exceeded`. Результат — постмортем, архивный тег, старт
следующей ветки по детерминированной таблице переходов.

---

## Предусловия

- Запись `{event:"gate_eval", result:"fail", ...}` **уже** в `log.jsonl`.
  Либо запись `{event:"budget_extended"}` была отклонена, и активен
  неявный `budget_exceeded`.
- Не существует записи `{event:"backtrack", branch:<id>}` для этой
  попытки (повторный backtrack одной и той же ветки запрещён;
  используется следующий резерв).

## Алгоритм

1. **Определить следующий `branch_id`** по §8 ROOT.md.
   - Если строки нет или `next="—"` → `meta_backtrack` (§10).
   - Если `next` уже в архиве (есть тег `medp/archive/<next>-fail-*`) →
     взять следующую строку того же исходного `on_fail_next` ряда
     (А2→A6 при занятом A3 и т.п.). Если ряд исчерпан → `meta_backtrack`.

2. **Собрать postmortem.** Создать `branches/<id>/postmortem.md` с жёстко
   заданными секциями:

   ```
   # Postmortem: <branch_id>

   ## Причина fail
   gate=<gate_id>, value=<v>, threshold=<t>, comparator=<cmp>.

   ## Гипотеза о причине (1–3 пункта)
   ...

   ## Артефакты, полезные для других ветвей
   (из поля preserve_on_fail в plan.md; перечислить пути)

   ## Что НЕ делать в <next>
   (1–3 антипаттерна, выведенных из этой попытки)

   ## Что стоит попробовать в <next>
   (1–3 конкретных отличия от текущей ветки)
   ```

3. **Архивировать.** Не удалять ветку. Создать тег:
   `git tag medp/archive/<branch_id>-fail-<gate_id>`.

4. **Записать событие backtrack в `log.jsonl`:**
   ```
   {"ts":"<ISO>","event":"backtrack","branch":"<branch_id>","author":"<who>","reason":"<gate_id>-fail","next":"<next_id>"}
   ```

5. **Коммит.** Точный формат:
   `medp: backtrack <branch_id> reason=<gate_id>-fail next=<next_id>`.

6. **Старт следующей ветки** по `prompts/00_kickoff.md` + §9.1 ROOT.md.
   Использовать `preserve_on_fail`-артефакты прошлой ветки, если они
   релевантны (указать это явно в `branches/<next>/plan.md`).

## Запреты

- Нельзя пропустить `postmortem.md`. Без него коммит backtrack запрещён.
- Нельзя пересматривать пороги гейтов провалившейся ветки ретроактивно.
  Пересмотр возможен только **до** старта ветки через
  `threshold_adjusted` с обоснованием.
- Нельзя удалять ветку `medp/<branch_id>` и тег `medp/archive/*`.

## Рамка ответа

```
## Backtrack branch=<branch_id>

- Причина: gate=<gate_id>, fail (value=<v>, threshold=<t>, cmp=<cmp>).
- Следующая ветка по §8 ROOT.md: <next_id>.
  Обоснование выбора: <одна фраза или "прямое правило таблицы">.

## Сгенерированный postmortem.md

<полный текст, готов к записи в branches/<branch_id>/postmortem.md>

## Записи

log.jsonl append:
<JSON-строка>

commit message:
<строка>

tag:
<строка>

## Предлагаемый переход к prompts/00_kickoff.md для старта <next_id>.

## Ожидаю подтверждения перед исполнением.
```
