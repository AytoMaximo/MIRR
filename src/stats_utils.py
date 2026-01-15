"""stats_utils.py — минимально достаточные статистические примитивы.

Назначение
— Глава 4 использует «контроль риска» через биномиальные доли (ложные принятия
  на негативных попытках). Для редких событий и ограниченных выборок точечная
  оценка x/N систематически занижает риск, поэтому в калибровке используется
  верхняя доверительная граница.

Принятая конвенция
— beta_ci — уровень значимости β для односторонней границы, т.е. верхняя
  граница UB(x,N;1-β) гарантирует покрытие не хуже 1-β.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist


def clopper_pearson_ub(x: int, n: int, beta_ci: float = 0.05) -> float:
    """Односторонняя верхняя граница Clopper–Pearson для доли в Bin(n,p).

    Возвращает UB(x,n;1-β), где β = beta_ci.

    Формула:
        UB = 1, если x = n;
        UB = Beta^{-1}(1-β; x+1, n-x), иначе.

    Здесь Beta^{-1} — квантиль бета-распределения.

    Ссылочный факт: Clopper–Pearson — «точный» интервал для биномиальной доли,
    основанный на инверсии хвостовых тестов.
    """
    n = int(n)
    x = int(x)
    if n <= 0:
        return float("nan")
    if x < 0 or x > n:
        raise ValueError(f"x должно быть в [0,n], получено x={x}, n={n}")
    beta_ci = float(beta_ci)
    if not (0.0 < beta_ci < 1.0):
        raise ValueError("beta_ci должно быть в (0,1)")

    if x == n:
        return 1.0
    # scipy.stats.beta.ppf корректно обрабатывает x=0
    ub = float(beta_dist.ppf(1.0 - beta_ci, x + 1, n - x))
    # страховка от численных артефактов
    return float(np.clip(ub, 0.0, 1.0))
