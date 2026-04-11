from __future__ import annotations
import argparse
import os
from typing import List, Tuple, Optional
import numpy as np

from io_utils import read_csv_two_columns
from interpolation import DividedDifferences, FiniteDifferences, lagrange_interpolate
from plotting import plot_interpolation, plot_abs_error


def resample_linear(x: List[float], y: List[float], m: int) -> Tuple[List[float], List[float]]:
    """
    Створює m рівномірних вузлів на [min(x), max(x)] і задає y лінійною інтерполяцією між точками.
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
    headers = ["i", "x_i"] + [f"col{j}" for j in range(n)]
    print(" | ".join(h.rjust(10) for h in headers))
    print("-" * (13 * (len(headers))))
    for i in range(n):
        row = [str(i), f"{x[i]:.6g}"]
        for j in range(n):
            v = table[i][j]
            row.append("" if v is None else f"{v:.6g}")
        print(" | ".join(c.rjust(10) for c in row))


def compute_once(
    x: List[float],
    y: List[float],
    x_star: float,
    save_dir: Optional[str] = None,
    label: str = "",
    save_plots: bool = False,
) -> None:
    pairs = sorted(zip(x, y), key=lambda t: t[0])
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    print(f"\n[{label}] Вузлів: {len(x)}; x*: {x_star}\n")

    # Ньютон
    dd = DividedDifferences.build(x, y)
    print_table(f"[{label}] Таблиця розділених різниць (Ньютон):", x, dd.as_table())
    p_newton = dd.evaluate(x_star)
    print(f"[{label}] P_Newton({x_star}) = {p_newton:.10g}")

    # Факторіальний (Δ) — якщо рівномірні вузли
    p_fact = None
    fd = None
    try:
        fd = FiniteDifferences.build_uniform(x, y)
        print_table(f"[{label}] Таблиця скінченних різниць (Δ):", x, fd.as_table())
        p_fact = fd.evaluate_forward(x_star)
        print(f"[{label}] P_Factorial/Forward({x_star}) = {p_fact:.10g}")
    except Exception as e:
        print(f"[{label}] [УВАГА] Factorial через Δ не застосовано: {e}")

    # Лагранж
    p_lagr = lagrange_interpolate(x, y, x_star)
    print(f"[{label}] P_Lagrange({x_star}) = {p_lagr:.10g}")

    # Функції
    fN = lambda xx: dd.evaluate(float(xx))
    fL = lambda xx: lagrange_interpolate(x, y, float(xx))
    fF = (lambda xx: fd.evaluate_forward(float(xx))) if fd is not None else None

    x_min, x_max = min(x), max(x)

    if save_plots:
        if save_dir is None:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)

        plot_interpolation(
            x, y, fN,
            title=f"Ньютон (розділені різниці) {label}",
            out_png=os.path.join(save_dir, f"plot_newton_{label}.png"),
        )
        plot_interpolation(
            x, y, fL,
            title=f"Лагранж {label}",
            out_png=os.path.join(save_dir, f"plot_lagrange_{label}.png"),
        )
        plot_abs_error(
            (x_min, x_max), fN, fL,
            title=f"|Newton - Lagrange| {label}",
            out_png=os.path.join(save_dir, f"plot_err_newton_vs_lagrange_{label}.png"),
        )

        if fF is not None:
            plot_interpolation(
                x, y, fF,
                title=f"Factorial (Δ) {label}",
                out_png=os.path.join(save_dir, f"plot_factorial_{label}.png"),
            )
            plot_abs_error(
                (x_min, x_max), fN, fF,
                title=f"|Newton - Factorial| {label}",
                out_png=os.path.join(save_dir, f"plot_err_newton_vs_factorial_{label}.png"),
            )

    print(f"\n[{label}] ПІДСУМОК")
    print(f"[{label}] Newton:   {p_newton:.10g}")
    if p_fact is not None:
        print(f"[{label}] Factorial:{p_fact:.10g}")
    else:
        print(f"[{label}] Factorial: (не застосовано)")
    print(f"[{label}] Lagrange: {p_lagr:.10g}")
    print("-" * 70)


def parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Вхідний CSV з 2 колонками: x,y")
    ap.add_argument("--x", type=float, required=True, help="Точка прогнозу x*")
    ap.add_argument("--save_plots", action="store_true", help="Зберегти графіки у PNG")
    ap.add_argument("--study", default="", help="Список кількостей вузлів, напр. 5,10,20 (для дослідження)")
    ap.add_argument("--study_mode", choices=["resample", "subset"], default="resample",
                    help="Як отримувати більше вузлів: resample (лінійно) або subset (взяти перші k)")
    ap.add_argument("--out_dir", default="outputs_lab2", help="Папка для графіків/результатів")
    args = ap.parse_args()

    x, y = read_csv_two_columns(args.csv)

    if args.study.strip():
        ks = parse_int_list(args.study)
        os.makedirs(args.out_dir, exist_ok=True)

        for k in ks:
            if args.study_mode == "subset":
                xk, yk = subset_nodes(x, y, k)
            else:
                xk, yk = resample_linear(x, y, k)

            label = f"k{k}"
            compute_once(
                xk, yk, args.x,
                save_dir=args.out_dir,
                label=label,
                save_plots=args.save_plots,
            )
        return

    compute_once(
        x, y, args.x,
        save_dir=args.out_dir,
        label="single",
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()