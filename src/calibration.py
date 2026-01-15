
"""
calibration.py — вспомогательные процедуры для подбора порогов (глава 4.3).

Идея: по уже собранному журналу попыток "проиграть" альтернативные пороги
принятия (τ_min, Δτ_min) как дополнительный фильтр и сравнить компромисс
между доступностью (доля принятий вовремя) и риском неверного принятия.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from metrics import AnchorTolerances, succ_within_time, derive_hazard
from stats_utils import clopper_pearson_ub


@dataclass(frozen=True)
class SweepConfig:
    # требование к доступности (минимальная доля принятий вовремя)
    p_accept_min: float
    # маршрутный риск-бюджет (доля неблагоприятных исходов на маршрут)
    alpha_route: float
    # эффективное число попыток на маршрут (целое)
    n_eff: int
    # уровень значимости β для односторонней верхней границы UB(·;1-β)
    beta_ci: float = 0.05


def _pass_min_gate(df: pd.DataFrame, tau_min: float, delta_tau_min: float) -> pd.Series:
    tau1 = df["tau1"].astype(float)
    delta_tau = df["delta_tau"].astype(float)
    has_dt = np.isfinite(delta_tau)
    # если маржинальность (Δτ) не определена, используем только τ(1)
    pass_tau = (tau1 >= float(tau_min)) & (~has_dt | (delta_tau >= float(delta_tau_min)))

    # gate — наблюдаемый бинарный результат дополнительных проверок допустимости (0/1).
    # Если поле отсутствует (реальные логи могут быть минимальными), считаем gate=1.
    if "gate" in df.columns:
        gate_ok = df["gate"].astype(int) == 1
    else:
        gate_ok = pd.Series(True, index=df.index)
    return pass_tau & gate_ok


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


def calibrate_thresholds_operational_loso(
    attempts: pd.DataFrame,
    *,
    time_budget_s: float,
    tau_min_grid: Iterable[float],
    delta_tau_min_grid: Iterable[float],
    cfg: SweepConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Эксплуатационная калибровка по негативным попыткам с разбиением по сессиям.

    Идея (исправление замечаний §4.3):
    1) Данные делятся по сессиям (LOSO): одна сессия — тестовая, остальные — обучающие.
    2) На обучающих сессиях подбираются пороги, которые удовлетворяют ограничению
       по верхней доверительной границе вероятности ложного принятия на негативных
       попытках: UB(x_neg, N_neg; 1-β) ≤ α_step.
    3) Отчётные метрики доступности и риска считаются только на отложенной сессии.

    Ожидается наличие колонки is_positive (1 — «позитивная попытка», 0 — «негативная»).
    Если колонки нет, функция возвращает пустые таблицы.
    """
    if "is_positive" not in attempts.columns:
        empty = pd.DataFrame()
        return empty, empty

    df = attempts.copy()
    df["succ"] = succ_within_time(df, float(time_budget_s))
    sessions = sorted(df["session_id"].unique().tolist())

    n_eff = int(max(1, cfg.n_eff))
    alpha_step = float(cfg.alpha_route) / float(n_eff)
    beta_ci = float(cfg.beta_ci)

    fold_rows: List[dict] = []
    for s_test in sessions:
        g_test = df[df["session_id"] == s_test]
        g_train = df[df["session_id"] != s_test]

        best: Optional[dict] = None
        # перебор кандидатов по train
        for tau_min in tau_min_grid:
            for delta_tau_min in delta_tau_min_grid:
                gate = _pass_min_gate(g_train, float(tau_min), float(delta_tau_min))
                accept_train = g_train["succ"] & gate

                neg_train = g_train[g_train["is_positive"].astype(int) == 0]
                if len(neg_train) == 0:
                    continue
                gate_neg = _pass_min_gate(neg_train, float(tau_min), float(delta_tau_min))
                accept_neg = succ_within_time(neg_train, float(time_budget_s)) & gate_neg
                x = int(np.sum(accept_neg))
                N = int(len(neg_train))
                ub = clopper_pearson_ub(x, N, beta_ci=beta_ci)
                if not (ub <= alpha_step):
                    continue

                pos_train = g_train[g_train["is_positive"].astype(int) == 1]
                if len(pos_train) == 0:
                    continue
                gate_pos = _pass_min_gate(pos_train, float(tau_min), float(delta_tau_min))
                accept_pos = succ_within_time(pos_train, float(time_budget_s)) & gate_pos
                p_accept_pos = float(np.mean(accept_pos))
                if p_accept_pos < float(cfg.p_accept_min):
                    continue

                cand = dict(
                    tau_min=float(tau_min),
                    delta_tau_min=float(delta_tau_min),
                    N_neg_train=N,
                    x_FA_train=x,
                    UB_FA_train=ub,
                    P_accept_pos_train=p_accept_pos,
                )
                if (best is None) or (cand["P_accept_pos_train"] > best["P_accept_pos_train"]):
                    best = cand

        # если ни одна настройка не прошла ограничения, всё равно пишем fold-строку (для диагностики)
        if best is None:
            fold_rows.append(dict(
                session_test=int(s_test),
                tau_min=np.nan,
                delta_tau_min=np.nan,
                alpha_step=alpha_step,
                beta_ci=beta_ci,
                N_neg_train=int(np.sum(g_train["is_positive"].astype(int) == 0)),
                x_FA_train=np.nan,
                UB_FA_train=np.nan,
                P_accept_pos_train=np.nan,
                N_neg_test=int(np.sum(g_test["is_positive"].astype(int) == 0)),
                x_FA_test=np.nan,
                FA_rate_test=np.nan,
                P_accept_pos_test=np.nan,
                ok="нет",
            ))
            continue

        # оценка на тестовой сессии
        tau_min = best["tau_min"]
        delta_tau_min = best["delta_tau_min"]

        neg_test = g_test[g_test["is_positive"].astype(int) == 0]
        gate_neg_t = _pass_min_gate(neg_test, float(tau_min), float(delta_tau_min)) if len(neg_test) else pd.Series([], dtype=bool)
        accept_neg_t = succ_within_time(neg_test, float(time_budget_s)) & gate_neg_t if len(neg_test) else pd.Series([], dtype=bool)
        x_t = int(np.sum(accept_neg_t)) if len(neg_test) else 0
        N_t = int(len(neg_test))
        fa_rate_t = float(x_t) / float(N_t) if N_t > 0 else np.nan

        pos_test = g_test[g_test["is_positive"].astype(int) == 1]
        gate_pos_t = _pass_min_gate(pos_test, float(tau_min), float(delta_tau_min)) if len(pos_test) else pd.Series([], dtype=bool)
        accept_pos_t = succ_within_time(pos_test, float(time_budget_s)) & gate_pos_t if len(pos_test) else pd.Series([], dtype=bool)
        p_accept_pos_t = float(np.mean(accept_pos_t)) if len(pos_test) else np.nan

        fold_rows.append(dict(
            session_test=int(s_test),
            tau_min=float(tau_min),
            delta_tau_min=float(delta_tau_min),
            alpha_step=float(alpha_step),
            beta_ci=float(beta_ci),
            N_neg_train=int(best["N_neg_train"]),
            x_FA_train=int(best["x_FA_train"]),
            UB_FA_train=float(best["UB_FA_train"]),
            P_accept_pos_train=float(best["P_accept_pos_train"]),
            N_neg_test=int(N_t),
            x_FA_test=int(x_t),
            FA_rate_test=float(fa_rate_t) if np.isfinite(fa_rate_t) else np.nan,
            P_accept_pos_test=float(p_accept_pos_t) if np.isfinite(p_accept_pos_t) else np.nan,
            ok="да",
        ))

    folds = pd.DataFrame(fold_rows)

    # агрегированная сводка по результатам LOSO
    ok_folds = folds[folds["ok"] == "да"]
    if len(ok_folds) == 0:
        summary = pd.DataFrame([dict(
            n_folds=int(len(folds)),
            n_ok_folds=0,
            tau_min_mode=np.nan,
            delta_tau_min_mode=np.nan,
            P_accept_pos_test_mean=np.nan,
            FA_rate_test_mean=np.nan,
        )])
    else:
        # наиболее частые пороги среди успешных фолдов
        mode = (
            ok_folds.groupby(["tau_min", "delta_tau_min"]).size().reset_index(name="count")
            .sort_values("count", ascending=False).iloc[0]
        )
        summary = pd.DataFrame([dict(
            n_folds=int(len(folds)),
            n_ok_folds=int(len(ok_folds)),
            tau_min_mode=float(mode["tau_min"]),
            delta_tau_min_mode=float(mode["delta_tau_min"]),
            P_accept_pos_test_mean=float(ok_folds["P_accept_pos_test"].mean()),
            FA_rate_test_mean=float(ok_folds["FA_rate_test"].mean()),
        )])

    return folds, summary


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


def make_pretty_loso(folds: pd.DataFrame) -> pd.DataFrame:
    """Читаемый вариант таблицы LOSO-калибровки (для вставки в текст)."""
    if folds is None or folds.empty:
        return pd.DataFrame()
    out = folds.copy()
    out = out.rename(columns={
        "session_test": "Тестовая сессия",
        "tau_min": "Порог τ_min, доля",
        "delta_tau_min": "Порог Δτ_min, доля",
        "alpha_step": "Целевой риск на попытку α_step, доля",
        "beta_ci": "Уровень значимости β (односторонний)",
        "N_neg_train": "Негативных попыток (train), шт",
        "x_FA_train": "Ложных принятия (train), шт",
        "UB_FA_train": "Верхняя граница риска UB (train), доля",
        "P_accept_pos_train": "Доля своевременных принятий на позитиве (train), доля",
        "N_neg_test": "Негативных попыток (test), шт",
        "x_FA_test": "Ложных принятия (test), шт",
        "FA_rate_test": "Доля ложных принятия на негативе (test), доля",
        "P_accept_pos_test": "Доля своевременных принятий на позитиве (test), доля",
        "ok": "Ограничения выполнены",
    })
    return out
