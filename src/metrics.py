
"""
metrics.py — расчёт метрик из §3.3–§3.4 (глава 3) по журналам попыток и узлов.

Фокус: воспроизводимый расчёт метрик верхнего уровня над "чёрным ящиком" SLAM/VIO.
Используемые формулы: (3.31)–(3.49), (3.50)–(3.66), (3.71)–(3.76) из главы 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnchorTolerances:
    """Навигационные допуски якоря (см. ai = (T_AM, E_i, δ_i, φ_i, ver(i)) в (3.12))."""
    delta_m: float  # δ_i, м
    phi_rad: float  # φ_i, рад


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Пороги принятия/уверенности (см. (3.17)–(3.18), (3.28))."""
    tau_min: float          # τ_min
    delta_tau_min: float    # Δτ_min
    tau_conf: float         # τ_conf
    delta_tau_conf: float   # Δτ_conf


def _is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def succ_within_time(df: pd.DataFrame, time_budget_s: float) -> pd.Series:
    """Событие успеха восстановления за Time (3.23) для каждой попытки (строка df)."""
    t_acc = df["t_acc"].astype(float)
    t_start = df["t_start"].astype(float)
    return np.isfinite(t_acc) & ((t_acc - t_start) <= float(time_budget_s))


def accepted_by_deadline(df: pd.DataFrame, deadline_s: float) -> pd.Series:
    """Событие Succi,s,j≤t (3.24), где t = deadline_s (абсолютное время в сессии)."""
    t_acc = df["t_acc"].astype(float)
    return np.isfinite(t_acc) & (t_acc <= float(deadline_s))


def derive_confidence(df: pd.DataFrame, thr: AcceptanceThresholds) -> pd.Series:
    """Событие высокой уверенности Conf (пример (3.28)): τ(1) ≥ τ_conf и Δτ ≥ Δτ_conf (если есть Δτ)."""
    tau1 = df["tau1"].astype(float)
    delta_tau = df["delta_tau"].astype(float)
    has_dt = np.isfinite(delta_tau)
    # режим одной гипотезы: считаем, что маржинальность не определена -> только τ(1) ≥ τ_conf
    conf = (tau1 >= thr.tau_conf) & (~has_dt | (delta_tau >= thr.delta_tau_conf))
    return conf


def derive_hazard(df: pd.DataFrame, tol: Dict[int, AnchorTolerances]) -> pd.Series:
    """Событие Haz (пример (3.27)) по офлайн-ошибкам при наличии эталона (A3)."""
    err_plan = df["err_plan"].astype(float)
    err_yaw = df["err_yaw"].astype(float)
    anchor_id = df["anchor_id"].astype(int)

    delta = anchor_id.map(lambda i: tol[int(i)].delta_m).astype(float)
    phi = anchor_id.map(lambda i: tol[int(i)].phi_rad).astype(float)

    # модуль по yaw — как оговорено после (3.27)
    return (np.isfinite(err_plan) & (err_plan > delta)) | (np.isfinite(err_yaw) & (np.abs(err_yaw) > phi))


def derive_hazard_obs(
    df: pd.DataFrame,
    *,
    delta_tau_max: float,
    q_track_min: float | None = None,
    eta_min: float | None = None,
) -> pd.Series:
    """Событие Haz^{obs} — наблюдаемый суррогат навигационной опасности.

    Определение (прикладная версия для гл. 4.3).
    Haz^{obs}=1 фиксирует «подозрительное» принятие, которое можно выявить без
    эталона, опираясь только на поля журнала.

    Минимально достаточный компонент (устойчивый к разным платформам):
    - низкая маржинальность гипотез Δτ (неуверенная развязка между 1-й и 2-й
      гипотезами). В демо используем порог Δτ < delta_tau_max.

    Дополнительные (опциональные) компоненты, если они логируются:
    - низкое качество трекинга q_track < q_track_min;
    - низкая сохранность признаков η < eta_min.

    Замечание. Если Δτ не определена (например, платформа выдает одну гипотезу),
    считаем это неблагоприятным признаком и помечаем Haz^{obs}=1.
    """
    # Δτ — основной наблюдаемый суррогат (двухгипотезный режим).
    dt = df["delta_tau"].astype(float) if "delta_tau" in df.columns else pd.Series(np.nan, index=df.index)
    obs = (~np.isfinite(dt)) | (np.isfinite(dt) & (dt < float(delta_tau_max)))

    # дополнительные сигналы (если заданы)
    if q_track_min is not None:
        q = df["q_track"].astype(float) if "q_track" in df.columns else pd.Series(np.nan, index=df.index)
        obs = obs | (np.isfinite(q) & (q < float(q_track_min)))
    if eta_min is not None:
        eta = df["eta"].astype(float) if "eta" in df.columns else pd.Series(np.nan, index=df.index)
        obs = obs | (np.isfinite(eta) & (eta < float(eta_min)))
    return obs


def validate_haz_obs_against_haz(
    attempts: pd.DataFrame,
    *,
    accept_mask: pd.Series,
    tol: Dict[int, AnchorTolerances],
    delta_tau_max: float,
    q_track_min: float | None,
    eta_min: float | None,
) -> pd.DataFrame:
    """Мини-эксперимент сопоставления Haz^{obs} и Haz на данных с эталоном.

    Возвращает одну строку с условными вероятностями и частотами:
    - Pr(Haz=1 | Haz^{obs}=1)
    - Pr(Haz=1 | Haz^{obs}=0)
    а также TPR/FPR для бинарного предиката Haz^{obs}.
    """
    df = attempts.copy()
    haz_true = derive_hazard(df, tol)
    haz_obs = derive_hazard_obs(
        df,
        delta_tau_max=float(delta_tau_max),
        q_track_min=(float(q_track_min) if q_track_min is not None else None),
        eta_min=(float(eta_min) if eta_min is not None else None),
    )

    # интересуют именно случаи, когда попытка прошла фильтры и была «принята»
    accept_mask = accept_mask.astype(bool)
    haz_true_acc = (accept_mask & haz_true)
    haz_obs_acc = (accept_mask & haz_obs)

    # условные вероятности
    n_obs1 = int(haz_obs_acc.sum())
    n_obs0 = int((accept_mask & (~haz_obs)).sum())
    p_haz_given_obs1 = float((haz_true_acc & haz_obs).sum()) / float(n_obs1) if n_obs1 > 0 else float("nan")
    p_haz_given_obs0 = float((haz_true_acc & (~haz_obs)).sum()) / float(n_obs0) if n_obs0 > 0 else float("nan")

    # TPR/FPR
    n_haz = int(haz_true_acc.sum())
    n_nonhaz = int((accept_mask & (~haz_true)).sum())
    tpr = float((haz_true_acc & haz_obs).sum()) / float(n_haz) if n_haz > 0 else float("nan")
    fpr = float(((~haz_true) & haz_obs_acc).sum()) / float(n_nonhaz) if n_nonhaz > 0 else float("nan")

    return pd.DataFrame([dict(
        delta_tau_max=float(delta_tau_max),
        q_track_min=(float(q_track_min) if q_track_min is not None else np.nan),
        eta_min=(float(eta_min) if eta_min is not None else np.nan),
        N_accept=int(accept_mask.sum()),
        N_haz=int(n_haz),
        N_haz_obs=int(n_obs1),
        P_haz_given_haz_obs1=p_haz_given_obs1,
        P_haz_given_haz_obs0=p_haz_given_obs0,
        TPR=tpr,
        FPR=fpr,
    )])


def compute_tail_probabilities(df: pd.DataFrame, time_budget_s: float, tol: Dict[int, AnchorTolerances]) -> pd.DataFrame:
    """Pitail,δ(Time) и Pitail,φ(Time) условно на успех (3.36)–(3.37)."""
    df = df.copy()
    df["succ"] = succ_within_time(df, time_budget_s)
    out = []
    for i, g in df.groupby("anchor_id"):
        g_succ = g[g["succ"]]
        if len(g_succ) == 0:
            out.append(dict(anchor_id=int(i), tail_plan=np.nan, tail_yaw=np.nan))
            continue
        d = tol[int(i)].delta_m
        ph = tol[int(i)].phi_rad
        tail_plan = float(np.mean(g_succ["err_plan"].astype(float) > d))
        tail_yaw = float(np.mean(np.abs(g_succ["err_yaw"].astype(float)) > ph))
        out.append(dict(anchor_id=int(i), tail_plan=tail_plan, tail_yaw=tail_yaw))
    return pd.DataFrame(out)


def var_cvar(values: np.ndarray, alpha: float) -> Tuple[float, float]:
    """VaR_α и CVaR_α по (3.41)–(3.42) для выборки values (без NaN)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    # эмпирический квантиль: используем "higher", чтобы соответствовать inf{c: P(C<=c) >= alpha}
    var = float(np.quantile(v, alpha, method="higher"))
    tail = v[v >= var]
    cvar = float(np.mean(tail)) if tail.size > 0 else float(var)
    return var, cvar


def compute_anchor_table(
    attempts: pd.DataFrame,
    time_budget_s: float,
    tol: Dict[int, AnchorTolerances],
    thr: AcceptanceThresholds,
    alpha_cvar: float,
    lambda_yaw_m_per_rad: float,
) -> pd.DataFrame:
    """
    Таблица метрик по якорям:
    - P̂_succ(Time) (3.32)
    - P̂_conf(Time)=Pr(ConfErr ∧ Succ(Time)) (3.38) — эмпирическая доля попыток
    - хвостовые вероятности превышения допусков (3.36)–(3.37) условно на успех
    - VaR/CVaR по стоимости (3.41)–(3.42) условно на успех
    - E[T_acc | Succ(Time)] (3.44)
    """
    df = attempts.copy()
    df["succ"] = succ_within_time(df, time_budget_s)
    df["conf_derived"] = derive_confidence(df, thr)
    df["haz_derived"] = derive_hazard(df, tol)
    df["conferr_derived"] = df["conf_derived"] & df["haz_derived"] & df["succ"]

    # стоимость C (3.40): e_plan + λ |e_yaw|
    df["C"] = df["err_plan"].astype(float) + float(lambda_yaw_m_per_rad) * np.abs(df["err_yaw"].astype(float))

    out_rows = []
    for i, g in df.groupby("anchor_id"):
        i = int(i)
        n = len(g)
        p_succ = float(np.mean(g["succ"])) if n > 0 else np.nan
        p_conferr = float(np.mean(g["conferr_derived"])) if n > 0 else np.nan

        g_succ = g[g["succ"]]
        if len(g_succ) > 0:
            tail_plan = float(np.mean(g_succ["err_plan"].astype(float) > tol[i].delta_m))
            tail_yaw = float(np.mean(np.abs(g_succ["err_yaw"].astype(float)) > tol[i].phi_rad))
            var_a, cvar_a = var_cvar(g_succ["C"].to_numpy(), alpha_cvar)
            t_acc = g_succ["t_acc"].astype(float) - g_succ["t_start"].astype(float)
            e_tacc = float(np.mean(t_acc))  # (3.44)
            med_tacc = float(np.median(t_acc))
        else:
            tail_plan = tail_yaw = var_a = cvar_a = e_tacc = med_tacc = np.nan

        out_rows.append(dict(
            anchor_id=i,
            N_attempts=n,
            P_succ=p_succ,
            P_conferr=p_conferr,
            tail_plan=tail_plan,
            tail_yaw=tail_yaw,
            VaR_alpha=var_a,
            CVaR_alpha=cvar_a,
            E_Tacc_s=e_tacc,
            Med_Tacc_s=med_tacc,
        ))

    return pd.DataFrame(out_rows).sort_values("anchor_id").reset_index(drop=True)


def p_succ_curve(attempts: pd.DataFrame, time_grid_s: Iterable[float]) -> pd.DataFrame:
    """P̂_succ(Time) как функция Time для каждого якоря (3.32)."""
    df = attempts.copy()
    rows = []
    for T in time_grid_s:
        succ = succ_within_time(df, float(T))
        tmp = df.assign(succ=succ).groupby("anchor_id")["succ"].mean().reset_index()
        tmp["Time_s"] = float(T)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True).sort_values(["anchor_id","Time_s"])


def route_alpha_step(alpha_route: float, n_eff: int) -> float:
    """Калибровка α_step из (3.47) при фиксированном n_eff (попытки как проверки)."""
    n_eff = int(max(1, n_eff))
    return float(alpha_route) / float(n_eff)


def classify_failures(attempts: pd.DataFrame, time_budget_s: float) -> pd.Series:
    """
    Инженерная классификация попыток (гл. 4.4).

    Важно: значения меток делаем человекочитаемыми (с пробелами),
    чтобы их можно было напрямую использовать в таблицах и на диаграммах
    без «программистских» подчёркиваний.
    """
    df = attempts.copy()
    service = df["service_state"].astype(str)
    t_acc = df["t_acc"].astype(float)
    t_start = df["t_start"].astype(float)
    dt = t_acc - t_start

    succ = succ_within_time(df, time_budget_s)
    # берём готовые поля conferr/haz если есть, иначе ожидаем derive_* вне
    conferr = df["conferr"].astype(int) == 1 if "conferr" in df.columns else pd.Series(False, index=df.index)
    haz = df["haz"].astype(int) == 1 if "haz" in df.columns else pd.Series(False, index=df.index)

    out = []
    for idx in df.index:
        if service.iloc[idx] != "OK":
            out.append("сервис недоступен")
            continue
        if not np.isfinite(t_acc.iloc[idx]):
            out.append("не принял")
            continue
        if dt.iloc[idx] > float(time_budget_s):
            out.append("не успел")
            continue
        if conferr.iloc[idx]:
            out.append("уверенно неверно")
        elif haz.iloc[idx]:
            out.append("принял неверно")
        else:
            out.append("успех")
    return pd.Series(out, index=df.index, name="failure_class")


def compute_navigation_metrics(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Метрики уровня навигации по узлам (3.59)–(3.66) на уровне эмпирических частот.
    Ожидается, что в nodes есть столбцы: session_id, node_id, ready, turn_ok, node_fail, w_m.
    """
    df = nodes.copy()
    df["ready"] = df["ready"].astype(int)
    df["turn_ok"] = df["turn_ok"].astype(int)
    df["node_fail"] = df["node_fail"].astype(int)
    df["w_m"] = df["w_m"].astype(float)

    # по узлам: вероятность Ready и условная вероятность неверного выбора при Ready
    out_rows = []
    for m, g in df.groupby("node_id"):
        m = int(m)
        p_ready = float(np.mean(g["ready"]))
        g_ready = g[g["ready"] == 1]
        p_wrong_given_ready = float(np.mean(1 - g_ready["turn_ok"])) if len(g_ready) else np.nan
        p_fail = float(np.mean(g["node_fail"]))
        out_rows.append(dict(node_id=m, P_ready=p_ready, P_wrong_given_ready=p_wrong_given_ready, P_node_fail=p_fail))
    out_nodes = pd.DataFrame(out_rows).sort_values("node_id")

    # по маршруту: Cost_nav (3.66) и вероятность хотя бы одного отказа на сессию
    df["cost_nav"] = df["w_m"] * df["node_fail"]
    cost_by_session = df.groupby("session_id")["cost_nav"].sum().reset_index()
    cost_by_session["any_fail"] = (cost_by_session["cost_nav"] > 0).astype(int)

    route_summary = pd.DataFrame([dict(
        E_Cost_nav=float(cost_by_session["cost_nav"].mean()),
        P_any_fail=float(cost_by_session["any_fail"].mean()),
        N_sessions=int(cost_by_session.shape[0]),
    )])

    return out_nodes, route_summary


def two_scale_degradation_detector(
    attempts: pd.DataFrame,
    alpha_fast: float,
    alpha_slow: float,
    theta_low: float,
    theta_gap: float,
    L: int,
) -> pd.DataFrame:
    """
    Двухмасштабный детектор деградации по (3.74)–(3.76).
    Вход: attempts с колонками session_id, anchor_id, eta (η_i,s,j).
    Выход: по (i,s) значения y_i,s, m_fast, m_slow и бинарный флаг degrade.
    """
    df = attempts.copy()
    df = df[np.isfinite(df["eta"].astype(float))]
    if df.empty:
        return pd.DataFrame(columns=["anchor_id","session_id","y","m_fast","m_slow","degrade"])

    # y_i,s = median_j eta_i,s,j (3.74)
    y = df.groupby(["anchor_id","session_id"])["eta"].median().reset_index().rename(columns={"eta":"y"})
    y = y.sort_values(["anchor_id","session_id"])

    rows = []
    for i, g in y.groupby("anchor_id"):
        g = g.sort_values("session_id").reset_index(drop=True)
        m_fast = None
        m_slow = None
        degrade_flags = []
        # EMA: m_s = α m_{s-1} + (1-α) y_s (3.75)
        for idx, r in g.iterrows():
            if m_fast is None:
                m_fast = float(r["y"])
                m_slow = float(r["y"])
            else:
                m_fast = float(alpha_fast) * m_fast + (1.0 - float(alpha_fast)) * float(r["y"])
                m_slow = float(alpha_slow) * m_slow + (1.0 - float(alpha_slow)) * float(r["y"])
            rows.append(dict(anchor_id=int(i), session_id=int(r["session_id"]), y=float(r["y"]),
                             m_fast=m_fast, m_slow=m_slow))
        # деградация: m_slow < theta_low и |m_fast-m_slow| <= theta_gap на протяжении L сессий (3.76)
    out = pd.DataFrame(rows).sort_values(["anchor_id","session_id"]).reset_index(drop=True)
    if out.empty:
        out["degrade"] = False
        return out

    out["degrade"] = False
    for i, g in out.groupby("anchor_id"):
        idxs = g.index.to_list()
        cond = (g["m_slow"] < float(theta_low)) & (np.abs(g["m_fast"] - g["m_slow"]) <= float(theta_gap))
        # скользящее окно длины L
        for k in range(len(idxs)):
            if k + int(L) <= len(idxs):
                if bool(cond.iloc[k:k+int(L)].all()):
                    out.loc[idxs[k:k+int(L)], "degrade"] = True
    return out
