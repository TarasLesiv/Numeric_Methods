from typing import List, Tuple
import matplotlib.pyplot as plt

from lsq import FitResult


def plot_variances(results: List[FitResult]) -> None:
    degrees = [r.degree for r in results]
    vars_ = [r.var for r in results]

    plt.figure()
    plt.plot(degrees, vars_, marker="o")
    plt.title("Залежність дисперсії від степеня полінома")
    plt.xlabel("Степінь m")
    plt.ylabel("Дисперсія")
    plt.grid(True)


def plot_approximation(x: List[float], y: List[float], y_hat: List[float], m: int) -> None:
    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-", label="Фактичні дані")
    plt.plot(x, y_hat, marker=".", linestyle="--", label=f"Апроксимація (m={m})")
    plt.title("Апроксимація температури методом МНК")
    plt.xlabel("Місяць")
    plt.ylabel("Температура")
    plt.grid(True)
    plt.legend()


def plot_error(err_table: List[Tuple[float, float]], m: int) -> None:
    xs = [t[0] for t in err_table]
    es = [t[1] for t in err_table]

    plt.figure()
    plt.plot(xs, es, marker="o")
    plt.title(f"Похибка апроксимації (m={m}), err = y - ŷ")
    plt.xlabel("Місяць")
    plt.ylabel("Похибка")
    plt.grid(True)