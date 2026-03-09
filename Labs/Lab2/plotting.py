from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_interpolation(
    x: List[float],
    y: List[float],
    f_interp: Callable[[float], float],
    title: str,
    out_png: Optional[str] = None,
    f_true: Optional[Callable[[float], float]] = None,
) -> None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    x_dense = np.linspace(np.min(x_arr), np.max(x_arr), 600)
    y_dense = np.array([f_interp(float(xx)) for xx in x_dense], dtype=float)

    plt.figure()
    plt.scatter(x_arr, y_arr, label="вузли (дані)")
    plt.plot(x_dense, y_dense, label="інтерполяція")

    if f_true is not None:
        y_true = np.array([f_true(float(xx)) for xx in x_dense], dtype=float)
        plt.plot(x_dense, y_true, label="істинна функція")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_abs_error(
    x_range: Tuple[float, float],
    f_a: Callable[[float], float],
    f_b: Callable[[float], float],
    title: str,
    out_png: Optional[str] = None,
) -> None:
    x_dense = np.linspace(x_range[0], x_range[1], 600)
    err = np.array([abs(f_a(float(xx)) - f_b(float(xx))) for xx in x_dense], dtype=float)

    plt.figure()
    plt.plot(x_dense, err)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("|f_a - f_b|")
    plt.grid(True)

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()