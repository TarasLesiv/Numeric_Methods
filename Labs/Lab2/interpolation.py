from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import math
import numpy as np


@dataclass
class DividedDifferences:
    """
    Таблиця розділених різниць для інтерполяції Ньютона.
    dd[i][j] = f[x_i, ..., x_{i+j}]
    """
    x: np.ndarray
    dd: np.ndarray  # upper-triangular (n x n)

    @staticmethod
    def build(x: List[float], y: List[float]) -> "DividedDifferences":
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(x_arr)
        if n != len(y_arr):
            raise ValueError("x та y повинні мати однакову довжину")
        if n < 2:
            raise ValueError("Потрібно щонайменше 2 вузли")

        dd = np.zeros((n, n), dtype=float)
        dd[:, 0] = y_arr
        for j in range(1, n):
            for i in range(0, n - j):
                denom = x_arr[i + j] - x_arr[i]
                if denom == 0:
                    raise ValueError("Знайдено повторювані x — розділені різниці не визначені")
                dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / denom
        return DividedDifferences(x=x_arr, dd=dd)

    def coeffs(self) -> np.ndarray:
        """Коефіцієнти Ньютона: a0=dd[0,0], a1=dd[0,1], ..., a_{n-1}=dd[0,n-1]."""
        return self.dd[0, :].copy()

    def evaluate(self, x_star: float) -> float:
        """P(x*) у формі Ньютона (послідовне нарощування)."""
        a = self.coeffs()
        n = len(self.x)
        p = a[0]
        mult = 1.0
        for k in range(1, n):
            mult *= (x_star - self.x[k - 1])
            p += a[k] * mult
        return float(p)

    def as_table(self) -> List[List[Optional[float]]]:
        """Зручний табличний вигляд для виводу (None для порожніх місць)."""
        n = len(self.x)
        table: List[List[Optional[float]]] = []
        for i in range(n):
            row: List[Optional[float]] = []
            for j in range(n):
                if i + j < n:
                    row.append(float(self.dd[i, j]))
                else:
                    row.append(None)
            table.append(row)
        return table


@dataclass
class FiniteDifferences:
    """
    Таблиця скінченних різниць для рівномірної сітки.
    delta[i][j] = Δ^j y_i
    """
    x0: float
    h: float
    delta: np.ndarray  # upper-triangular (n x n)
    y0: float

    @staticmethod
    def build_uniform(x: List[float], y: List[float], tol: float = 1e-9) -> "FiniteDifferences":
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(x_arr)
        if n != len(y_arr):
            raise ValueError("x та y повинні мати однакову довжину")
        if n < 2:
            raise ValueError("Потрібно щонайменше 2 вузли")

        # перевірка рівномірності кроку
        steps = np.diff(x_arr)
        h = float(steps[0])
        if not np.all(np.abs(steps - h) <= tol * max(1.0, abs(h))):
            raise ValueError("Вузли не рівновіддалені — факторіальні многочлени через Δ не застосовні напряму")

        delta = np.zeros((n, n), dtype=float)
        delta[:, 0] = y_arr
        for j in range(1, n):
            for i in range(0, n - j):
                delta[i, j] = delta[i + 1, j - 1] - delta[i, j - 1]
        return FiniteDifferences(x0=float(x_arr[0]), h=h, delta=delta, y0=float(y_arr[0]))

    def evaluate_forward(self, x_star: float) -> float:
        """
        Інтерполяція вперед за факторіальним рядом (Ньютона вперед) через Δ:
        P(x) = y0 + p Δy0 + p(p-1)/2! Δ^2 y0 + ...
        де p = (x-x0)/h
        """
        p = (x_star - self.x0) / self.h
        n = self.delta.shape[0]
        res = self.delta[0, 0]
        falling = 1.0  # p(p-1)...
        fact = 1.0
        for k in range(1, n):
            falling *= (p - (k - 1))
            fact *= k
            res += (falling / fact) * self.delta[0, k]
        return float(res)

    def as_table(self) -> List[List[Optional[float]]]:
        n = self.delta.shape[0]
        table: List[List[Optional[float]]] = []
        for i in range(n):
            row: List[Optional[float]] = []
            for j in range(n):
                if i + j < n:
                    row.append(float(self.delta[i, j]))
                else:
                    row.append(None)
            table.append(row)
        return table


def lagrange_interpolate(x: List[float], y: List[float], x_star: float) -> float:
    """Класична формула Лагранжа (для порівняння)."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)
    total = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if i == j:
                continue
            denom = x_arr[i] - x_arr[j]
            if denom == 0:
                raise ValueError("Повторювані x у Лагранжі")
            li *= (x_star - x_arr[j]) / denom
        total += y_arr[i] * li
    return float(total)
