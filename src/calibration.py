
"""
calibration.py — вспомогательные процедуры для подбора порогов (глава 4.3).

Идея: по уже собранному журналу попыток "проиграть" альтернативные пороги
принятия (τ_min, Δτ_min) как дополнительный фильтр и сравнить компромисс
между доступностью (доля принятий вовремя) и риском неверного принятия.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from metrics import AnchorTolerances, succ_within_time, derive_hazard


@dataclass(frozen=True)
class SweepConfig:
    # требование к доступности (минимальная доля принятий вовремя)
    p_accept_min: float
    # маршрутный риск-бюджет (доля неблагоприятных исходов на маршрут)
    alpha_route: float
    # эффективное число попыток на маршрут (целое)
    n_eff: int


def _pass_min_gate(df: pd.DataFrame, tau_min: float, delta_tau_min: float) -> pd.Series:
    tau1 = df["tau1"].astype(float)
    delta_tau = df["delta_tau"].astype(float)
    has_dt = np.isfinite(delta_tau)
    # если маржинальность (Δτ) не определена, используем только τ(1)
    return (tau1 >= float(tau_min)) & (~has_dt | (delta_tau >= float(delta_tau_min)))


def sweep_thresholds(
    attempts: pd.DataFrame,
    *,
    time_budget_s: float,
    tol: Dict[int, AnchorTolerances],
    tau_min_grid: Iterable[float],
    delta_tau_min_grid: Iterable[float],
    cfg: SweepConfig,
) -> pd.DataFrame:
    """
    Таблица перебора порогов:
    - выбираем якорь, который проходит ограничение по риску и даёт максимум P_accept;
    - проверяем ограничения: риск на шаг <= alpha_route / n_eff и P_accept >= p_accept_min.
    """
    df = attempts.copy()
    df["succ"] = succ_within_time(df, float(time_budget_s))
    df["haz"] = derive_hazard(df, tol)

    n_eff = int(max(1, cfg.n_eff))
    alpha_step = float(cfg.alpha_route) / float(n_eff)

    rows = []
    for tau_min in tau_min_grid:
        for delta_tau_min in delta_tau_min_grid:
            gate = _pass_min_gate(df, float(tau_min), float(delta_tau_min))
            accept = df["succ"] & gate

            per_anchor = []
            for anchor_id, g in df.groupby("anchor_id"):
                idx = g.index
                acc = accept.loc[idx]
                p_accept = float(np.mean(acc)) if len(g) else np.nan
                p_wrong = float(np.mean(acc & g["haz"])) if len(g) else np.nan
                per_anchor.append((int(anchor_id), p_accept, p_wrong))

            # выбираем якорь с максимальным P_accept среди тех, кто проходит риск
            feasible = [(aid, pa, pw) for (aid, pa, pw) in per_anchor if np.isfinite(pw) and (pw <= alpha_step)]
            if len(feasible) == 0:
                chosen = None
                p_best = np.nan
                pw_best = np.nan
                ok_risk = False
            else:
                ok_risk = True
                chosen, p_best, pw_best = sorted(feasible, key=lambda x: x[1], reverse=True)[0]

            ok_avail = bool(ok_risk and np.isfinite(p_best) and (p_best >= float(cfg.p_accept_min)))
            rows.append(dict(
                tau_min=float(tau_min),
                delta_tau_min=float(delta_tau_min),
                n_eff=n_eff,
                alpha_step=alpha_step,
                anchor_chosen=(int(chosen) if chosen is not None else np.nan),
                P_accept_best=p_best,
                P_wrong_best=pw_best,
                ok_flag=(1 if ok_avail else 0),
                ok=("да" if ok_avail else "нет"),
            ))

    out = pd.DataFrame(rows)
    out = out.sort_values(["ok_flag", "P_accept_best"], ascending=[False, False]).reset_index(drop=True)
    return out


def make_pretty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ok_flag" in out.columns:
        out = out.drop(columns=["ok_flag"])
    out = out.rename(columns={
        "tau_min": "Порог τ_min, доля",
        "delta_tau_min": "Порог Δτ_min, доля",
        "n_eff": "Эффективное число попыток, шт",
        "alpha_step": "Допустимая доля неверных принятия на попытку, доля",
        "anchor_chosen": "Выбранный якорь",
        "P_accept_best": "Доля своевременных принятий (для выбранного якоря), доля",
        "P_wrong_best": "Доля неверных принятия (для выбранного якоря), доля",
        "ok": "Ограничения выполнены",
    })
    return out
