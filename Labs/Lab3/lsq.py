from dataclasses import dataclass
from typing import List, Tuple

from linalg import gauss_solve


def form_matrix_A(x: List[float], m: int) -> List[List[float]]:
    n = m + 1
    A = [[0.0 for _ in range(n)] for _ in range(n)]

    power_sums = [0.0 for _ in range(2 * m + 1)]
    for xi in x:
        val = 1.0
        for p in range(0, 2 * m + 1):
            power_sums[p] += val
            val *= xi

    for i in range(n):
        for j in range(n):
            A[i][j] = power_sums[i + j]
    return A


def form_vector_b(x: List[float], y: List[float], m: int) -> List[float]:
    n = m + 1
    b = [0.0 for _ in range(n)]
    for xi, yi in zip(x, y):
        val = 1.0
        for i in range(n):
            b[i] += yi * val
            val *= xi
    return b


def poly_values(xs: List[float], coef: List[float]) -> List[float]:
    res = []
    for x in xs:
        val = 0.0
        p = 1.0
        for c in coef:
            val += c * p
            p *= x
        res.append(val)
    return res


def variance(y_true: List[float], y_approx: List[float]) -> float:
    if len(y_true) != len(y_approx):
        raise ValueError("Різні розміри масивів для variance")
    s = 0.0
    for a, b in zip(y_true, y_approx):
        d = a - b
        s += d * d
    return s / len(y_true)


def error_table(x: List[float], y: List[float], y_hat: List[float]) -> List[Tuple[float, float]]:
    return [(xi, yi - yhi) for xi, yi, yhi in zip(x, y, y_hat)]


@dataclass
class FitResult:
    degree: int
    coef: List[float]
    var: float


def fit_degree(x: List[float], y: List[float], m: int) -> FitResult:
    A = form_matrix_A(x, m)
    b = form_vector_b(x, y, m)
    coef = gauss_solve(A, b)
    y_hat = poly_values(x, coef)
    var = variance(y, y_hat)
    return FitResult(degree=m, coef=coef, var=var)


def best_fit(x: List[float], y: List[float], m_min: int = 1, m_max: int = 4) -> Tuple[List[FitResult], FitResult]:
    results = [fit_degree(x, y, m) for m in range(m_min, m_max + 1)]
    best = min(results, key=lambda r: r.var)
    return results, best


def forecast_next_months(x: List[float], coef: List[float], months_ahead: int = 3) -> Tuple[List[float], List[float]]:
    last = max(x)
    x_future = [last + i for i in range(1, months_ahead + 1)]
    y_future = poly_values(x_future, coef)
    return x_future, y_future