# Rust-порт / semiotic_hypercube: индекс

Точка входа в документацию neuro-symbolic Rust-крейта и его
эволюционно-исследовательского протокола.

## Операционный слой (MEDP)

- `medp/ROOT.md` — корневой план: словарь, форки A1..A3 + резерв A4..A12,
  сетка гейтов G1..G6, таблица переходов, meta-backtrack.
- `medp/log.jsonl` — append-only журнал событий MEDP (bootstrap / gate /
  backtrack / fork / cleanup).
- `medp/prompts/` — типовые промпты для возобновления работы:
  - `00_kickoff.md`        — восстановление контекста новой сессии.
  - `01_evaluate_gate.md`  — оценка чекпоинта и commit-формат.
  - `02_backtrack.md`      — постмортем + откат на запасной форк.
  - `03_fork.md`           — добавление новой развилки в резерв.
- `medp/branches/` — рабочие каталоги активных MEDP-веток (A1/, A2/, …).

## Архитектура / теория

- `SEMIOTIC-HYPERCUBE.md` — концепт-документ (придуман с Gemini).
  Neuro-symbolic слой поверх GGGP-двигателя.
- `../rust/tests/ontological_crucible.rs` — intergration-тест инвариантов
  (dimensional integrity, fractal determinism, hybrid convergence).
- `../rust/tests/python_bridge/README.md` — how-to для PyO3 bridge-теста.

## Связь с экосистемой

- `gggp` (PyPI, <https://pypi.org/project/gggp/>, owner: olkhovoy) —
  **foundational engine**: Grammar-Guided Genetic Programming как
  standalone Rust + Python SDK.
- `semiotic_hypercube` (этот крейт) — **research / application layer**
  поверх gggp: векторный фенотип, фрактальная декодировка, CMA-ES,
  Semiotic Closure Loop.

При переносе `gggp_bundle/` в отдельный репозиторий `semiotic_hypercube`
оба уровня остаются разделёнными: gggp — библиотека, semiotic_hypercube —
исследовательский надслой.
