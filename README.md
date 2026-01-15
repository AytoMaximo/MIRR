
# Воспроизводимый расчёт метрик и графики

Это минимальный пакет для главы 4: он показывает, как из журнала попыток восстановления (окна `W_{i,s,j}`) и (опционально) журнала узлов выбора считать метрики из §3.3–§3.4.

## Быстрый старт (демо)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/run_all.py
```

Результаты:
- `out/tables/anchor_metrics.csv` — таблица метрик по якорям.
- `out/tables/p_succ_curve.csv` — кривая P̂_succ(T).
- `out/figures/p_succ_vs_T.png` — график P̂_succ(T).
- `out/figures/err_plan_hist.png` — распределение ошибок при успехе.
- `out/figures/alpha_step_vs_n_eff.png` — накопление маршрутного риска (α_step).
- `out/tables/degradation_detector.csv` — пример двухмасштабного детектора деградации.

## Подстановка реальных логов

См. `data/README.md` и `src/io_schema.md`.
