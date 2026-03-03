from __future__ import annotations
import argparse
from typing import List, Tuple, Optional
import numpy as np

from io_utils import read_csv_two_columns, write_tabulation
from interpolation import DividedDifferences, FiniteDifferences, lagrange_interpolate
from plotting import plot_interpolation, plot_abs_error


def resample_linear(x: List[float], y: List[float], m: int) -> Tuple[List[float], List[float]]:
    """
    Створює m рівномірних вузлів на [min(x), max(x)] і задає y лінійною інтерполяцією між точками.
    УВАГА: це синтетичні точки (не експеримент).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    idx = np.argsort(x_arr)
    x_arr = x_arr[idx]
    y_arr = y_arr[idx]

    x_new = np.linspace(float(x_arr[0]), float(x_arr[-1]), m)
    y_new = np.interp(x_new, x_arr, y_arr)
    return x_new.tolist(), y_new.tolist()


def subset_nodes(x: List[float], y: List[float], k: int) -> Tuple[List[float], List[float]]:
    if k >= len(x):
        return x, y
    return x[:k], y[:k]


def print_table(title: str, x: List[float], table: List[List[Optional[float]]]) -> None:
    print("\n" + title)
    n = len(x)
    # Заголовок
    headers = ["i", "x_i"] + [f"col{j}" for j in range(n)]
    print(" | ".join(h.rjust(10) for h in headers))
    print("-" * (13 * (len(headers))))
    for i in range(n):
        row = [str(i), f"{x[i]:.6g}"]
        for j in range(n):
            v = table[i][j]
            row.append("" if v is None else f"{v:.6g}")
        print(" | ".join(c.rjust(10) for c in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Вхідний CSV з 2 колонками: x,y")
    ap.add_argument("--x", type=float, required=True, help="Точка прогнозу (x*)")
    ap.add_argument("--method", choices=["newton", "factorial", "lagrange", "both"], default="both")
    ap.add_argument("--k", type=int, default=0, help="Використати лише перші k вузлів (0 = всі)")
    ap.add_argument("--resample", type=int, default=0, help="Створити m вузлів ресемплінгом (0 = не робити)")
    ap.add_argument("--save_plots", action="store_true", help="Зберегти графіки у PNG замість показу")
    args = ap.parse_args()

    x, y = read_csv_two_columns(args.csv)

    if args.k and args.k > 0:
        x, y = subset_nodes(x, y, args.k)

    if args.resample and args.resample > 0:
        x, y = resample_linear(x, y, args.resample)

    # Сортуємо вузли (важливо!)
    pairs = sorted(zip(x, y), key=lambda t: t[0])
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    print(f"Вузлів: {len(x)}; x*: {args.x}\n")

    # 1) Ньютон (розділені різниці)
    dd = DividedDifferences.build(x, y)
    print_table("Таблиця розділених різниць (Ньютон):", x, dd.as_table())
    p_newton = dd.evaluate(args.x)
    print(f"\nP_Newton({args.x}) = {p_newton:.10g}")

    # 2) Факторіальний (скінченні різниці) — тільки якщо рівномірна сітка
    p_fact = None
    fd = None
    try:
        fd = FiniteDifferences.build_uniform(x, y)
        print_table("Таблиця скінченних різниць (Δ):", x, fd.as_table())
        p_fact = fd.evaluate_forward(args.x)
        print(f"\nP_Factorial/Forward({args.x}) = {p_fact:.10g}")
    except Exception as e:
        print("\n[УВАГА] Факторіальний метод через Δ не застосовано:", str(e))

    # 3) Лагранж (для порівняння)
    p_lagr = lagrange_interpolate(x, y, args.x)
    print(f"P_Lagrange({args.x}) = {p_lagr:.10g}")

    # Функції для графіків
    fN = lambda xx: dd.evaluate(float(xx))
    fL = lambda xx: lagrange_interpolate(x, y, float(xx))
    if fd is not None:
        fF = lambda xx: fd.evaluate_forward(float(xx))
    else:
        fF = None

    x_min, x_max = min(x), max(x)

    # Графіки
    out1 = "plot_newton.png" if args.save_plots else None
    plot_interpolation(x, y, fN, "Інтерполяція Ньютона (розділені різниці)", out_png=out1)

    out2 = "plot_lagrange.png" if args.save_plots else None
    plot_interpolation(x, y, fL, "Інтерполяція Лагранжа", out_png=out2)

    if fF is not None:
        out3 = "plot_factorial.png" if args.save_plots else None
        plot_interpolation(x, y, fF, "Факторіальний (Ньютон вперед через Δ)", out_png=out3)
        out4 = "plot_err_newton_vs_factorial.png" if args.save_plots else None
        plot_abs_error((x_min, x_max), fN, fF, "Похибка |Newton - Factorial|", out_png=out4)

    out5 = "plot_err_newton_vs_lagrange.png" if args.save_plots else None
    plot_abs_error((x_min, x_max), fN, fL, "Похибка |Newton - Lagrange|", out_png=out5)

    # Підсумок
    print("\nПІДСУМОК")
    if args.method in ("newton", "both"):
        print(f"Newton:   {p_newton:.10g}")
    if args.method in ("factorial", "both"):
        if p_fact is not None:
            print(f"Factorial:{p_fact:.10g}")
        else:
            print("Factorial: (не застосовано — вузли не рівномірні)")
    if args.method in ("lagrange", "both"):
        print(f"Lagrange: {p_lagr:.10g}")


if __name__ == "__main__":
    main()
