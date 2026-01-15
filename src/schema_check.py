"""schema_check.py — проверки входных CSV перед расчётом.

Назначение: давать понятные инженерные ошибки вида
«в attempts.csv отсутствует столбец …», вместо падения внутри расчёта.

Проверки намеренно мягкие:
- лишние поля разрешены;
- некоторые поля (например, err_plan/err_yaw/eta) считаются опциональными;
  при отсутствии они добавляются как NaN, а метрики, зависящие от эталона,
  становятся неопределёнными (NaN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


_ATTEMPTS_REQUIRED = [
    "session_id",
    "anchor_id",
    "attempt_id",
    "t_start",
    "t_end",
    "t_acc",
    "service_state",
    "tau1",
]

_ATTEMPTS_OPTIONAL_WITH_NAN = [
    "tau2",
    "delta_tau",
    "gate",
    "q_track",
    "eta",
    "err_plan",
    "err_yaw",
    "conf",
    "haz",
    "conferr",
]


_NODES_REQUIRED = [
    "session_id",
    "node_id",
    "ready",
    "turn_ok",
    "node_fail",
    "w_m",
]


def _missing_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]


def validate_attempts(df: pd.DataFrame, *, file_label: str = "attempts.csv") -> pd.DataFrame:
    """Проверяет и нормализует журнал попыток.

    Возвращает копию df с добавленными опциональными колонками (NaN),
    а также с нормализованным t_acc (строка 'inf' -> np.inf).
    """
    miss = _missing_cols(df, _ATTEMPTS_REQUIRED)
    if miss:
        raise ValueError(f"В {file_label} отсутствуют обязательные столбцы: {', '.join(miss)}")

    out = df.copy()

    # добавляем опциональные столбцы
    for c in _ATTEMPTS_OPTIONAL_WITH_NAN:
        if c not in out.columns:
            out[c] = np.nan

    # нормализация t_acc: допускаем 'inf' как строку
    out["t_acc"] = out["t_acc"].replace({"inf": np.inf, "Inf": np.inf, "INF": np.inf})
    out["t_acc"] = pd.to_numeric(out["t_acc"], errors="coerce")

    # типы идентификаторов
    out["session_id"] = pd.to_numeric(out["session_id"], errors="raise").astype(int)
    out["anchor_id"] = pd.to_numeric(out["anchor_id"], errors="raise").astype(int)
    out["attempt_id"] = pd.to_numeric(out["attempt_id"], errors="raise").astype(int)

    # времена
    out["t_start"] = pd.to_numeric(out["t_start"], errors="raise").astype(float)
    out["t_end"] = pd.to_numeric(out["t_end"], errors="raise").astype(float)

    return out


def validate_nodes(df: pd.DataFrame, *, file_label: str = "nodes.csv") -> pd.DataFrame:
    """Проверяет и нормализует журнал узлов выбора."""
    miss = _missing_cols(df, _NODES_REQUIRED)
    if miss:
        raise ValueError(f"В {file_label} отсутствуют обязательные столбцы: {', '.join(miss)}")

    out = df.copy()
    out["session_id"] = pd.to_numeric(out["session_id"], errors="raise").astype(int)
    out["node_id"] = pd.to_numeric(out["node_id"], errors="raise").astype(int)
    out["ready"] = pd.to_numeric(out["ready"], errors="raise").astype(int)
    out["turn_ok"] = pd.to_numeric(out["turn_ok"], errors="raise").astype(int)
    out["node_fail"] = pd.to_numeric(out["node_fail"], errors="raise").astype(int)
    out["w_m"] = pd.to_numeric(out["w_m"], errors="raise").astype(float)
    return out
