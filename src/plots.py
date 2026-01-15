
"""
plots.py — графики для демонстрационного расчёта (matplotlib).

Требуемые графики:
1) P̂_succ(Time) как функция Time (для 1–2 якорей).
2) Распределение ошибок при успехе (гистограмма/ECDF).
3) Пример накопления маршрутного риска: α_step как функция n_eff при фиксированном α_route.
(опционально) диаграмма типов отказов.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import os

import numpy as np
import pandas as pd

# Matplotlib в headless-средах часто пытается писать кэш шрифтов/настроек в
# домашний каталог. Чтобы избежать подвисаний из-за прав/блокировок, принудительно
# задаём каталог конфигурации внутри проекта.
_mpl_cfg = Path(__file__).resolve().parents[1] / "out" / ".mplconfig"
_mpl_cfg.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg))

# Принудительно используем неинтерактивный backend.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_p_succ_curve(curve: pd.DataFrame, anchor_ids: Sequence[int], out_path: Path) -> None:
    _ensure_parent(out_path)
    plt.figure()
    for aid in anchor_ids:
        g = curve[curve["anchor_id"] == aid]
        plt.plot(g["Time_s"], g["succ"], marker="o", label=f"Якорь {aid}")
    # В подписях осей используем только сущности и единицы измерения (без формул).
    plt.xlabel("Бюджет времени, с")
    plt.ylabel("Доля своевременных восстановлений")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_error_distribution(attempts: pd.DataFrame, succ_mask: pd.Series, out_path: Path, kind: str = "hist") -> None:
    _ensure_parent(out_path)
    errs = attempts.loc[succ_mask, "err_plan"].astype(float)
    errs = errs[np.isfinite(errs)]
    if errs.empty:
        return

    plt.figure()
    if kind == "ecdf":
        x = np.sort(errs.to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post")
        plt.xlabel("Ошибка положения на плане, м")
        plt.ylabel("Накопленная доля, доля")
        plt.ylim(0.0, 1.05)
    else:
        plt.hist(errs.to_numpy(), bins=18)
        plt.xlabel("Ошибка положения на плане, м")
        plt.ylabel("Частота, шт")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_alpha_step(alpha_route: float, n_eff_grid: Iterable[int], out_path: Path) -> None:
    _ensure_parent(out_path)
    n = np.array(list(n_eff_grid), dtype=int)
    alpha_step = alpha_route / np.maximum(1, n)
    plt.figure()
    plt.plot(n, alpha_step, marker="o")
    plt.xlabel("Эффективное число попыток, шт")
    plt.ylabel("Целевой уровень риска на одну попытку, доля")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_failure_pie(attempts: pd.DataFrame, out_path: Path) -> None:
    _ensure_parent(out_path)
    if "failure_class" not in attempts.columns:
        return
    vc = attempts["failure_class"].value_counts()
    # На диаграмме используем человекочитаемые подписи (без подчёркиваний).
    labels = [str(x).replace("_", " ") for x in vc.index]
    plt.figure()
    plt.pie(vc.values, labels=labels, autopct="%1.0f%%")
    plt.title("Доли типов исходов попыток")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
