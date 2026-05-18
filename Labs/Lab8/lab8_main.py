import math
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

def f(x):
    return math.cos(x)

def df(x):
    return -math.sin(x)

def ddf(x):
    return -math.cos(x)

def check_stop(x_new, x_old, eps):
    return abs(f(x_new)) < eps and abs(x_new - x_old) < eps

def tabulate(a, b, h, filename="tabulation.txt"):
    x_vals, y_vals = [], []
    out_path = Path(filename)
    if not out_path.is_absolute():
        out_path = BASE_DIR / out_path

    with open(out_path, "w") as file:
        x = a
        while x <= b:
            y = f(x)
            file.write(f"{x:.4f}\t{y:.4f}\n")
            x_vals.append(x)
            y_vals.append(y)
            x += h
    return x_vals, y_vals

def eval_poly(coeffs, x):
    res = 0
    m = len(coeffs) - 1
    for i, a in enumerate(coeffs):
        res += a * (x ** (m - i))
    return res

def simple_iteration(x0, tau, eps):
    it = 0
    x_prev = x0
    while True:
        it += 1
        x_curr = x_prev + tau * f(x_prev)
        if check_stop(x_curr, x_prev, eps): return x_curr, it
        x_prev = x_curr

def newton_method(x0, eps):
    it = 0
    x_prev = x0
    while True:
        it += 1
        x_curr = x_prev - f(x_prev) / df(x_prev)
        if check_stop(x_curr, x_prev, eps): return x_curr, it
        x_prev = x_curr

def chebyshev_method(x0, eps):
    it = 0
    x_prev = x0
    while True:
        it += 1
        fn, dfn, ddfn = f(x_prev), df(x_prev), ddf(x_prev)
        x_curr = x_prev - fn / dfn - 0.5 * (fn ** 2 * ddfn) / (dfn ** 3)
        if check_stop(x_curr, x_prev, eps): return x_curr, it
        x_prev = x_curr

def secant_method(x_prev, x_curr, eps):
    it = 0
    while True:
        it += 1
        x_next = x_curr - f(x_curr) * (x_curr - x_prev) / (f(x_curr) - f(x_prev))
        if check_stop(x_next, x_curr, eps): return x_next, it
        x_prev, x_curr = x_curr, x_next

def parabolic_method(x0, x1, x2, eps):
    it = 0
    while True:
        it += 1
        f01 = (f(x1) - f(x0)) / (x1 - x0)
        f12 = (f(x2) - f(x1)) / (x2 - x1)
        f012 = (f12 - f01) / (x2 - x0)
        w = f12 + (x2 - x1) * f012
        det = math.sqrt(w**2 - 4 * f(x2) * f012)
        delta = -2 * f(x2) / (w + det if abs(w + det) > abs(w - det) else w - det)
        x_next = x2 + delta.real
        if abs(f(x_next)) < eps and abs(x_next - x2) < eps: return x_next, it
        x0, x1, x2 = x1, x2, x_next

def inverse_interpolation_3p(x0, x1, x2, eps):
    it = 0
    while True:
        it += 1
        y0, y1, y2 = f(x0), f(x1), f(x2)
        x_next = (y1*y2)/((y0-y1)*(y0-y2))*x0 + \
                 (y0*y2)/((y1-y0)*(y1-y2))*x1 + \
                 (y0*y1)/((y2-y0)*(y2-y1))*x2
        if abs(f(x_next)) < eps and abs(x_next - x2) < eps: return x_next, it
        x0, x1, x2 = x1, x2, x_next

def read_coeffs(filename):
    path = Path(filename)
    if not path.is_absolute():
        path = BASE_DIR / path
    with open(path, "r") as f:
        return [float(c) for c in f.read().split()]


def horner_newton(a, x0, eps):
    n = len(a) - 1
    x_curr = x0
    it = 0
    while True:
        it += 1

        b = a[0]
        c = 0.0
        for i in range(1, n + 1):
            c = c * x_curr + b
            b = b * x_curr + a[i]

        p_val = b
        dp_val = c
        if dp_val == 0:
            raise ZeroDivisionError(
                "Horner-Newton failed: derivative is zero at x = " + str(x_curr)
            )

        x_next = x_curr - p_val / dp_val
        if abs(x_next - x_curr) < eps:
            return x_next, it
        x_curr = x_next

def lin_method(a, alpha0, beta0, eps):
    m = len(a) - 1
    p, q = -2 * alpha0, alpha0 ** 2 + beta0 ** 2
    it = 0
    while True:
        it += 1
        b = [0] * (m + 1)
        b[m] = a[m]
        b[m - 1] = a[m - 1] - p * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = a[i] - p * b[i + 1] - q * b[i + 2]

        q_new = a[0] / b[2]
        p_new = (a[1] * b[2] - a[0] * b[3]) / (b[2] ** 2)

        alpha_new = -p_new / 2
        det = q_new - alpha_new ** 2

        beta_new = math.sqrt(abs(det))

        if abs(alpha_new - alpha0) < eps and abs(beta_new - beta0) < eps:
            return complex(alpha_new, beta_new), it

        alpha0, beta0, p, q = alpha_new, beta_new, p_new, q_new

def plot_graphs(x_tab, y_tab, coeffs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x_tab, y_tab, label='F(x) = cos(x)', color='blue')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_title("Трансцендентна функція (Пункт 1)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("F(x)")
    ax1.grid(True)
    ax1.legend()

    x_poly = [i * 0.1 for i in range(-20, 31)]  # Від -2 до 3
    y_poly = [eval_poly(coeffs, x) for x in x_poly]

    ax2.plot(x_poly, y_poly, label='Алгебраїчне рівняння', color='green')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_title("Алгебраїчний многочлен (Пункт 5)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("P(x)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    out_path = BASE_DIR / "graphs_lab8.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    EPS = 1e-10

    x_st = 0.7
    tau_val = -0.5
    x_vals, y_vals = tabulate(-2, 2, 0.1)
    print("Алгебраїчне рівняння: F(x) = 1x^3 - 3x^2 + 4x - 2")
    coeffs = read_coeffs("coefficients.txt")
    print("Коефіціїнти: ", coeffs,"\n")
    print("1. Табуляцію завершено (див. tabulation.txt)")

    plot_graphs(x_vals, y_vals, coeffs)

    x_start = 1.0
    print("РЕЗУЛЬТАТИ ДЛЯ ТРАНСЦЕНДЕНТНОГО РІВНЯННЯ:")
    res_si = simple_iteration(x_st, tau_val, EPS)
    print(f"1. Проста ітерація:        x = {abs(res_si[0]):.10f}, ітерацій: {res_si[1]}")

    res_n = newton_method(x_st, EPS)
    print(f"2. Метод Ньютона:          x = {res_n[0]:.10f}, ітерацій: {res_n[1]}")

    res_ch = chebyshev_method(x_st, EPS)
    print(f"3. Метод Чебишева:         x = {res_ch[0]:.10f}, ітерацій: {res_ch[1]}")

    res_sec = secant_method(0.6, 0.8, EPS)
    print(f"4. Метод хорд:             x = {res_sec[0]:.10f}, ітерацій: {res_sec[1]}")

    res_par = parabolic_method(0.5, 0.6, 0.7, EPS)
    print(f"5. Метод парабол:          x = {res_par[0]:.10f}, ітерацій: {res_par[1]}")

    res_inv = inverse_interpolation_3p(0.5, 0.6, 0.7, EPS)
    print(f"6. Зворотна інтерполяція:  x = {res_inv[0]:.10f}, ітерацій: {res_inv[1]}")

    print("-" * 50)
    print("РЕЗУЛЬТАТИ ДЛЯ АЛГЕБРАЇЧНОГО РІВНЯННЯ:")
    res_h = horner_newton(coeffs, 1.5, EPS)
    print(f"Дійсний корінь (Горнер):   x = {res_h[0]:.10f}, ітерацій: {res_h[1]}")

    res_l = lin_method(coeffs, 0.5, 0.5, EPS)
    print(f"Комплексний корінь (Лін):  x = {res_l[0]}, ітерацій: {res_l[1]}")

    print("\n3. Алгебраїчне рівняння (з coefficients.txt):")
    real_root, h_it = horner_newton(coeffs, 1.5, EPS)
    print(f"   Дійсний корінь (Схема Горнера): {real_root:.10f}, ітерацій: {h_it}")

    comp_root, l_it = lin_method(coeffs, 0.5, 0.5, EPS)
    print(f"   Комплексний корінь (Метод Ліна): {comp_root}, ітерацій: {l_it}")