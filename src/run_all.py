"""run_all.py — единая точка запуска демонстрационного расчёта.

Скрипт читает:
- журнал попыток восстановления (attempts.csv);
- (опционально) журнал узлов выбора (nodes.csv).

На выход пишет:
- таблицы в out/tables;
- рисунки в out/figures;
- config_used.json (параметры расчёта);
- run_summary.json и run_summary.csv (паспорт прогона).

Типовой запуск (демо-данные):
    python src/run_all.py

Запуск на своих данных:
    python src/run_all.py --attempts data/attempts.csv --nodes data/nodes.csv

Параметры можно переопределить через JSON-конфиг:
    python src/run_all.py --config configs/exp1.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from metrics import (
    AcceptanceThresholds,
    AnchorTolerances,
    compute_anchor_table,
    classify_failures,
    p_succ_curve,
    succ_within_time,
    compute_navigation_metrics,
    two_scale_degradation_detector,
    validate_haz_obs_against_haz,
)
from calibration import (
    SweepConfig,
    sweep_thresholds,
    make_pretty,
    calibrate_thresholds_operational_loso,
    make_pretty_loso,
)

from plots import (
    plot_p_succ_curve,
    plot_error_distribution,
    plot_alpha_step,
    plot_failure_pie,
)
from schema_check import validate_attempts, validate_nodes


ROOT = Path(__file__).resolve().parents[1]


def _deep_update(dst: dict, src: dict) -> dict:
    """Рекурсивное обновление словаря (src перезаписывает dst)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = _deep_update(dict(dst[k]), v)
        else:
            dst[k] = v
    return dst


def _default_cfg() -> dict:
    return {
        "Time_budget_s": 3.0,
        "alpha_cvar": 0.9,
        "lambda_yaw_m_per_rad": 0.6,
        "alpha_route": 0.01,
        "n_eff_demo": 12,
        # Параметры демонстрации калибровки §4.3.
        # Важно: в небольших демо-журналах число негативных попыток ограничено,
        # поэтому для иллюстрации доверительных границ используется более
        # мягкий маршрутный риск-бюджет, чем в «продакшн» примере.
        "calibration_demo": {
            "alpha_route": 0.30,
            "p_accept_min": 0.15,
            "beta_ci": 0.05,
        },
        # Пороговая версия наблюдаемого суррогата Haz^{obs} (см. §4.3).
        # В демо берём «низкую маржинальность» Δτ как основной индикатор,
        # а q_track/η оставляем опциональными.
        "haz_obs": {
            "delta_tau_max": 0.20,
            "q_track_min": None,
            "eta_min": None,
        },
        "thresholds": {
            "tau_min": 0.6,
            "delta_tau_min": 0.15,
            "tau_conf": 0.85,
            "delta_tau_conf": 0.25,
        },
        "tolerances": {
            "1": {"delta_m": 0.5, "phi_rad": 0.35},
            "2": {"delta_m": 0.6, "phi_rad": 0.4},
        },
        "detector": {
            "alpha_fast": 0.35,
            "alpha_slow": 0.80,
            "theta_low": 0.45,
            "theta_gap": 0.08,
            "L": 3,
        },
    }


def _load_cfg(cfg_path: Path | None) -> dict:
    cfg = _default_cfg()
    if cfg_path is None:
        return cfg
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Файл конфигурации должен содержать JSON-объект (словарь).")
    return _deep_update(cfg, data)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Расчёт метрик и построение диагностических графиков по журналам попыток и узлов.",
    )
    parser.add_argument(
        "--attempts",
        type=Path,
        default=ROOT / "demo" / "demo_log.csv",
        help="Путь к CSV с журналом попыток восстановления.",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=ROOT / "demo" / "demo_nodes.csv",
        help="Путь к CSV с журналом узлов выбора (опционально).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "out",
        help="Выходной каталог; внутри будут созданы out/tables и out/figures.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON-файл с параметрами эксперимента (опционально).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed генератора случайных чисел для воспроизводимости демо.",
    )
    return parser.parse_args()


def _write_run_summary(
    out_tables: Path,
    *,
    attempts: pd.DataFrame,
    nodes_used: bool,
    cfg: dict,
    attempts_path: Path,
    nodes_path: Path,
) -> None:
    """Пишет паспорт прогона (JSON + CSV с двумя колонками: metric,value)."""
    n_attempts = int(attempts.shape[0])
    n_sessions = int(attempts["session_id"].nunique()) if n_attempts else 0
    n_anchors = int(attempts["anchor_id"].nunique()) if n_attempts else 0

    succ = succ_within_time(attempts, float(cfg["Time_budget_s"]))
    p_succ_overall = float(np.mean(succ)) if n_attempts else float("nan")

    if "failure_class" in attempts.columns and n_attempts:
        vc = attempts["failure_class"].value_counts()
        counts = {str(k): int(v) for k, v in vc.items()}
        shares = {str(k): float(v) / float(n_attempts) for k, v in vc.items()}
    else:
        counts = {}
        shares = {}

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "attempts_csv": str(attempts_path),
            "nodes_csv": str(nodes_path) if nodes_used else None,
        },
        "counts": {
            "attempts": n_attempts,
            "sessions": n_sessions,
            "anchors": n_anchors,
        },
        "key_parameters": {
            "time_budget_s": float(cfg["Time_budget_s"]),
            "alpha_route": float(cfg["alpha_route"]),
            "alpha_cvar": float(cfg["alpha_cvar"]),
            "lambda_yaw_m_per_rad": float(cfg["lambda_yaw_m_per_rad"]),
        },
        "aggregates": {
            "share_success_within_time": p_succ_overall,
            "outcome_counts": counts,
            "outcome_shares": shares,
        },
    }

    (out_tables / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV в виде «метрика, значение» удобнее для вставок и быстрого просмотра.
    rows = []
    rows.append(("Путь к журналу попыток", str(attempts_path)))
    rows.append(("Путь к журналу узлов", str(nodes_path) if nodes_used else "—"))
    rows.append(("Число попыток", n_attempts))
    rows.append(("Число сессий", n_sessions))
    rows.append(("Число якорей", n_anchors))
    rows.append(("Бюджет времени", float(cfg["Time_budget_s"])))
    rows.append(("Доля успехов в пределах бюджета", p_succ_overall))

    for k, v in counts.items():
        rows.append((f"Исход: {k} (шт)", v))
    for k, v in shares.items():
        rows.append((f"Исход: {k} (доля)", v))

    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(out_tables / "run_summary.csv", index=False)


def _write_pretty_tables(out_tables: Path, *, anchor_table: pd.DataFrame, curve: pd.DataFrame, alpha_cvar: float) -> None:
    """Дополнительные версии таблиц с русскими заголовками для вставки в текст."""
    am = anchor_table.copy()
    am = am.rename(columns={
        "anchor_id": "Якорь",
        "N_attempts": "Число попыток, шт",
        "P_succ": "Доля своевременных восстановлений, доля",
        "P_conferr": "Доля «уверенно неверно», доля",
        "tail_plan": "Доля превышений допуска по положению, доля",
        "tail_yaw": "Доля превышений допуска по курсу, доля",
        "VaR_alpha": f"Квантиль стоимости (уровень {alpha_cvar:.2f}), м",
        "CVaR_alpha": f"Средняя стоимость в худшей части (уровень {alpha_cvar:.2f}), м",
        "E_Tacc_s": "Среднее время до принятия, с",
        "Med_Tacc_s": "Медиана времени до принятия, с",
    })
    am.to_csv(out_tables / "anchor_metrics_pretty.csv", index=False)

    c = curve.copy()
    c = c.rename(columns={
        "anchor_id": "Якорь",
        "Time_s": "Бюджет времени, с",
        "succ": "Доля своевременных восстановлений, доля",
    })
    c.to_csv(out_tables / "p_succ_curve_pretty.csv", index=False)


def main() -> None:
    args = _parse_args()

    # фиксируем seed, чтобы демо было воспроизводимым
    np.random.seed(int(args.seed))

    cfg = _load_cfg(args.config)

    out_tables = Path(args.out) / "tables"
    out_figs = Path(args.out) / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    attempts_raw = pd.read_csv(args.attempts)
    attempts = validate_attempts(attempts_raw, file_label=str(args.attempts))

    thr = AcceptanceThresholds(**cfg["thresholds"])
    tol = {int(k): AnchorTolerances(**v) for k, v in cfg["tolerances"].items()}

    # Вспомогательное событие «принято по заданным порогам» для валидации Haz^{obs}.
    # Используем те же фильтры, что и в переборе порогов: успех по времени + пороги + gate.
    def _accept_mask_for_thresholds(df: pd.DataFrame, tau_min: float, delta_tau_min: float) -> pd.Series:
        tau1 = df["tau1"].astype(float)
        delta_tau = df["delta_tau"].astype(float)
        has_dt = np.isfinite(delta_tau)
        pass_tau = (tau1 >= float(tau_min)) & (~has_dt | (delta_tau >= float(delta_tau_min)))
        gate_ok = (df["gate"].astype(int) == 1) if ("gate" in df.columns) else pd.Series(True, index=df.index)
        return succ_within_time(df, float(cfg["Time_budget_s"])) & pass_tau & gate_ok

    anchor_table = compute_anchor_table(
        attempts=attempts,
        time_budget_s=float(cfg["Time_budget_s"]),
        tol=tol,
        thr=thr,
        alpha_cvar=float(cfg["alpha_cvar"]),
        lambda_yaw_m_per_rad=float(cfg["lambda_yaw_m_per_rad"]),
    )
    anchor_table.to_csv(out_tables / "anchor_metrics.csv", index=False)

    # кривая «доля успехов» по сетке бюджетов времени
    time_grid = np.linspace(0.5, 6.0, 12)
    curve = p_succ_curve(attempts, time_grid_s=time_grid)
    curve.to_csv(out_tables / "p_succ_curve.csv", index=False)

    # классификация исходов попыток
    attempts["failure_class"] = classify_failures(attempts, time_budget_s=float(cfg["Time_budget_s"]))
    attempts.to_csv(out_tables / "attempts_with_failure_class.csv", index=False)

    # Мини-эксперимент: сопоставление Haz^{obs} и Haz (только на данных с эталоном).
    hazobs_cfg = cfg.get("haz_obs", {})
    accept_for_val = _accept_mask_for_thresholds(
        attempts,
        tau_min=float(cfg["thresholds"]["tau_min"]),
        delta_tau_min=float(cfg["thresholds"]["delta_tau_min"]),
    )
    hazobs_val = validate_haz_obs_against_haz(
        attempts,
        accept_mask=accept_for_val,
        tol=tol,
        delta_tau_max=float(hazobs_cfg.get("delta_tau_max", 0.20)),
        q_track_min=hazobs_cfg.get("q_track_min", None),
        eta_min=hazobs_cfg.get("eta_min", None),
    )
    hazobs_val.to_csv(out_tables / "haz_obs_validation.csv", index=False)
    hazobs_val.rename(columns={
        "delta_tau_max": "Порог Δτ_max для Haz^{obs}, доля",
        "q_track_min": "Порог q_track_min, доля",
        "eta_min": "Порог η_min, доля",
        "N_accept": "Принято попыток (по порогам), шт",
        "N_haz": "Опасных принятий Haz=1, шт",
        "N_haz_obs": "Срабатываний Haz^{obs}=1, шт",
        "P_haz_given_haz_obs1": "Pr(Haz=1 | Haz^{obs}=1)",
        "P_haz_given_haz_obs0": "Pr(Haz=1 | Haz^{obs}=0)",
        "TPR": "TPR",
        "FPR": "FPR",
    }).to_csv(out_tables / "haz_obs_validation_pretty.csv", index=False)

    # --- §4.3. Калибровка порогов ---
    # 1) режим оценки (c эталоном): перебор порогов по Haz
    # 2) режим эксплуатации: калибровка по негативным попыткам + LOSO по сессиям
    #
    # В демо используем n_eff как число узлов выбора (если они есть), иначе берём cfg["n_eff_demo"].
    if args.nodes is not None and Path(args.nodes).exists():
        n_eff = int(pd.read_csv(args.nodes)["node_id"].nunique())
    else:
        n_eff = int(cfg.get("n_eff_demo", 10))

    calib_demo = cfg.get("calibration_demo", {})
    sweep_cfg = SweepConfig(
        p_accept_min=float(calib_demo.get("p_accept_min", 0.15)),
        alpha_route=float(calib_demo.get("alpha_route", cfg["alpha_route"])),
        n_eff=n_eff,
        beta_ci=float(calib_demo.get("beta_ci", 0.05)),
    )
    tau_grid = [0.60, 0.70, 0.80, 0.85]
    dt_grid = [0.15, 0.20, 0.25, 0.30]
    sweep_gt = sweep_thresholds(
        attempts,
        time_budget_s=float(cfg["Time_budget_s"]),
        tol=tol,
        tau_min_grid=tau_grid,
        delta_tau_min_grid=dt_grid,
        cfg=sweep_cfg,
    )

    sweep_gt.to_csv(out_tables / "threshold_sweep_gt.csv", index=False)
    make_pretty(sweep_gt).to_csv(out_tables / "threshold_sweep_gt_pretty.csv", index=False)

    folds, folds_summary = calibrate_thresholds_operational_loso(
        attempts,
        time_budget_s=float(cfg["Time_budget_s"]),
        tau_min_grid=tau_grid,
        delta_tau_min_grid=dt_grid,
        cfg=sweep_cfg,
    )
    folds.to_csv(out_tables / "threshold_loso_folds.csv", index=False)
    make_pretty_loso(folds).to_csv(out_tables / "threshold_loso_folds_pretty.csv", index=False)
    folds_summary.to_csv(out_tables / "threshold_loso_summary.csv", index=False)

    # рисунки
    plot_p_succ_curve(curve=curve, anchor_ids=[1, 2], out_path=out_figs / "p_succ_vs_T.png")
    succ_mask = succ_within_time(attempts, float(cfg["Time_budget_s"]))
    plot_error_distribution(attempts, succ_mask, out_path=out_figs / "err_plan_hist.png", kind="hist")
    n_eff_grid = list(range(1, 31))
    plot_alpha_step(float(cfg["alpha_route"]), n_eff_grid, out_path=out_figs / "alpha_step_vs_n_eff.png")
    # Отдельный график для §4.3 с риск-бюджетом демонстрационной калибровки.
    plot_alpha_step(float(sweep_cfg.alpha_route), n_eff_grid, out_path=out_figs / "alpha_step_vs_n_eff_calib.png")
    plot_failure_pie(attempts, out_path=out_figs / "failure_types_pie.png")

    # навигационные метрики (если задан файл узлов и он существует)
    nodes_used = False
    if args.nodes is not None and Path(args.nodes).exists():
        nodes_raw = pd.read_csv(args.nodes)
        nodes = validate_nodes(nodes_raw, file_label=str(args.nodes))
        node_table, route_summary = compute_navigation_metrics(nodes)
        node_table.to_csv(out_tables / "nav_nodes_metrics.csv", index=False)
        route_summary.to_csv(out_tables / "nav_route_summary.csv", index=False)
        nodes_used = True

    # детектор деградации
    det_cfg = cfg["detector"]
    det = two_scale_degradation_detector(
        attempts=attempts,
        alpha_fast=float(det_cfg["alpha_fast"]),
        alpha_slow=float(det_cfg["alpha_slow"]),
        theta_low=float(det_cfg["theta_low"]),
        theta_gap=float(det_cfg["theta_gap"]),
        L=int(det_cfg["L"]),
    )
    det.to_csv(out_tables / "degradation_detector.csv", index=False)

    # сохраняем конфиг для воспроизводимости
    (out_tables / "config_used.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # паспорт прогона + «читаемые» таблицы
    _write_run_summary(
        out_tables,
        attempts=attempts,
        nodes_used=nodes_used,
        cfg=cfg,
        attempts_path=args.attempts,
        nodes_path=args.nodes,
    )
    _write_pretty_tables(out_tables, anchor_table=anchor_table, curve=curve, alpha_cvar=float(cfg["alpha_cvar"]))


if __name__ == "__main__":
    main()
