from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import os
import numpy as np

from interpolation import DividedDifferences
from plotting import plot_interpolation, plot_abs_error


def runge_function(x: float) -> float:
    return 1.0 / (1.0 + 25.0 * x * x)


def equidistant_nodes(a: float, b: float, n: int) -> List[float]:
    return np.linspace(a, b, n).tolist()



def build_newton_interpolant(x: List[float], y: List[float]) -> Callable[[float], float]:
    dd = DividedDifferences.build(x, y)
    return lambda xx: dd.evaluate(float(xx))


def run_runge_plots(
    n_list: Sequence[int] = (5, 10, 20),
    a: float = -1.0,
    b: float = 1.0,
    out_dir: str = "outputs_runge",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    f_true = runge_function

    for n in n_list:
        #рівновіддалені вузли
        x_eq = equidistant_nodes(a, b, n)
        y_eq = [f_true(xi) for xi in x_eq]
        fN_eq = build_newton_interpolant(x_eq, y_eq)

        plot_interpolation(
            x_eq, y_eq, fN_eq,
            title=f"Runge: рівновіддалені вузли, n={n}",
            out_png=os.path.join(out_dir, f"runge_equidistant_n{n}.png"),
            f_true=f_true,
        )
        plot_abs_error(
            (a, b), f_true, fN_eq,
            title=f"Runge похибка |f - P|, рівновіддалені, n={n}",
            out_png=os.path.join(out_dir, f"runge_err_equidistant_n{n}.png"),
        )



if __name__ == "__main__":
    run_runge_plots()