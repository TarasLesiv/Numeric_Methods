import math


def M(t: float) -> float:
    """Функція вологості ґрунту."""
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)


def dM_exact(t: float) -> float:
    """Точна похідна."""
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)


def central_diff(f, t: float, h: float) -> float:
    """Центральна різницева формула для першої похідної."""
    return (f(t + h) - f(t - h)) / (2 * h)


def runge_romberg(D_h: float, D_h2: float, p: int) -> float:
    """Уточнення за методом Рунге–Ромберга."""
    return D_h2 + (D_h2 - D_h) / (2 ** p - 1)


def aitken_refinement(D_h: float, D_h2: float, D_h4: float) -> float:
    """Уточнення за методом Ейткена."""
    denominator = D_h4 - 2 * D_h2 + D_h
    if abs(denominator) < 1e-15:
        raise ZeroDivisionError("Знаменник у формулі Ейткена занадто малий.")
    return D_h - ((D_h2 - D_h) ** 2) / denominator


def aitken_order(D_h: float, D_h2: float, D_h4: float) -> float:
    """Оцінка порядку точності за методом Ейткена."""
    numerator = abs(D_h4 - D_h2)
    denominator = abs(D_h2 - D_h)
    if denominator < 1e-15 or numerator < 1e-15:
        raise ZeroDivisionError("Недостатньо точні дані для оцінки порядку.")
    return math.log(numerator / denominator, 2)


def main() -> None:
    t0 = 1.0
    exact = dM_exact(t0)

    print(f"M'(1) точне = {exact:.15f}")
    print()

    h_values = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]#похибка для різних зн

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

    h = 0.01#умова лб
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

    # Метод Рунге–Ромберга
    p_theoretical = 2
    D_rr = runge_romberg(D_h, D_h2, p_theoretical)
    err_rr = abs(D_rr - exact)

    print("Метод Рунге–Ромберга:")
    print(f"D_RR = {D_rr:.15f}")
    print(f"Похибка RR = {err_rr:.15e}")
    print()

    # Метод Ейткена
    D_aitken = aitken_refinement(D_h, D_h2, D_h4)
    err_aitken = abs(D_aitken - exact)
    p_aitken = aitken_order(D_h, D_h2, D_h4)

    print("Метод Ейткена:")
    print(f"D* = {D_aitken:.15f}")
    print(f"Похибка Ейткена = {err_aitken:.15e}")
    print(f"Оцінка порядку точності p = {p_aitken:.6f}")
    print()

    print("Висновок:")
    if exact < 0:
        print(
            f"У точці t = {t0} вологість зменшується, "
            f"швидкість висихання ≈ {abs(exact):.6f} одиниця/одиницю часу."
        )
    else:
        print(
            f"У точці t = {t0} вологість зростає, "
            f"швидкість зміни ≈ {exact:.6f} одиниця/одиницю часу."
        )

    print(
        "Методи Рунге–Ромберга та Ейткена суттєво зменшують похибку "
        "порівняно з базовою центральною різницею."
    )


if __name__ == "__main__":
    main()