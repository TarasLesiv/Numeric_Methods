import math
import matplotlib.pyplot as plt
import numpy as np


def M(t: float) -> float:
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)


def dM_exact(t: float) -> float:
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)


def central_diff(f, t: float, h: float) -> float:
    return (f(t + h) - f(t - h)) / (2 * h)

#уточнення
def runge_romberg(D_h: float, D_h2: float, p: int) -> float:
    return D_h2 + (D_h2 - D_h) / (2 ** p - 1)

#уточнення
def aitken_refinement(D_h: float, D_h2: float, D_h4: float) -> float:
    denominator = D_h4 - 2 * D_h2 + D_h
    if abs(denominator) < 1e-15:
        raise ZeroDivisionError("Знаменник у формулі Ейткена занадто малий.")
    return D_h - ((D_h2 - D_h) ** 2) / denominator


def aitken_order(D_h: float, D_h2: float, D_h4: float) -> float:
    numerator = abs(D_h4 - D_h2)
    denominator = abs(D_h2 - D_h)
    if denominator < 1e-15 or numerator < 1e-15:
        raise ZeroDivisionError("Недостатньо точні дані для оцінки порядку.")
    return math.log(numerator / denominator, 2)

#рівняння дотичної в точці t0
def tangent_line(t: np.ndarray, t0: float, M0: float, dM0: float) -> np.ndarray:
    return M0 + dM0 * (t - t0)


def plot_function_and_tangent(t0: float) -> None:
    t_values = np.linspace(0, 20, 400)
    M_values = np.array([M(t) for t in t_values])

    M0 = M(t0)
    dM0 = dM_exact(t0)
    tangent_values = tangent_line(t_values, t0, M0, dM0)

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, M_values, label="M(t) = 50e^(-0.1t) + 5sin(t)")
    plt.plot(t_values, tangent_values, linestyle="--", label=f"Дотична в t0 = {t0}")
    plt.scatter([t0], [M0], label=f"Точка дотику ({t0}, {M0:.3f})")

    plt.title("Графік функції вологості M(t) та дотичної")
    plt.xlabel("t")
    plt.ylabel("M(t)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_error_vs_h(t0: float, exact: float, h_values: list[float]) -> None:
    errors = []

    for h in h_values:
        approx = central_diff(M, t0, h)
        err = abs(approx - exact)
        errors.append(err)

    plt.figure(figsize=(10, 6))
    plt.plot(h_values, errors, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Залежність похибки чисельного диференціювання від кроку h")
    plt.xlabel("h")
    plt.ylabel("Абсолютна похибка")
    plt.grid(True)
    plt.show()


def main() -> None:
    t0 = 1.0
    exact = dM_exact(t0)

    print(f"Точне значення M'(1) = {exact:.15f}")
    print()

    h_values = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]

    print("Таблиця дослідження похибки:")
    print(f"{'h':>10} {'D(h)':>22} {'Похибка':>22}")

    best_h = None
    best_err = float("inf")

    for h in h_values:
        approx = central_diff(M, t0, h)
        err = abs(approx - exact)
        print(f"{h:>10.5g} {approx:>22.15f} {err:>22.15e}")

        if err < best_err:
            best_err = err
            best_h = h

    print()
    print(f"Оптимальний крок серед перевірених: h = {best_h}")
    print(f"Мінімальна похибка: {best_err:.15e}")
    print()

    h = 0.01
    D_h = central_diff(M, t0, h)
    D_h2 = central_diff(M, t0, h / 2)
    D_h4 = central_diff(M, t0, h / 4)

    err_h = abs(D_h - exact)

    print("Обчислення для h = 0.01:")
    print(f"D(h)   = {D_h:.15f}")
    print(f"D(h/2) = {D_h2:.15f}")
    print(f"D(h/4) = {D_h4:.15f}")
    print(f"Похибка при h = 0.01: {err_h:.15e}")
    print()

    p_theoretical = 2
    D_rr = runge_romberg(D_h, D_h2, p_theoretical)
    err_rr = abs(D_rr - exact)

    print("Метод Рунге–Ромберга:")
    print(f"D_RR = {D_rr:.15f}")
    print(f"Похибка RR = {err_rr:.15e}")
    print()

    D_aitken = aitken_refinement(D_h, D_h2, D_h4)
    err_aitken = abs(D_aitken - exact)
    p_aitken = aitken_order(D_h, D_h2, D_h4)

    print("Метод Ейткена:")
    print(f"D* = {D_aitken:.15f}")
    print(f"Похибка Ейткена = {err_aitken:.15e}")
    print(f"Оцінка порядку точності p = {p_aitken:.6f}")
    print()

    if exact < 0:
        print(
            f"У точці t = {t0} вологість зменшується, "
            f"швидкість висихання ≈ {abs(exact):.6f} од./од. часу."
        )
    else:
        print(
            f"У точці t = {t0} вологість зростає, "
            f"швидкість зміни ≈ {exact:.6f} од./од. часу."
        )

    print(
        "Методи Рунге–Ромберга та Ейткена зменшують похибку "
        "порівняно з базовою центральною різницею."
    )

    plot_function_and_tangent(t0)
    plot_error_vs_h(t0, exact, h_values)


if __name__ == "__main__":
    main()