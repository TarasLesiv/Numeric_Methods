import math
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

def f(x):
    return x ** 2 - math.exp(-x)

def df(x):
    return 2 * x + math.exp(-x)

def ddf(x):
    return 2 - math.exp(-x)

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

    ax1.plot(x_tab, y_tab, label='F(x) = x^2 - exp(-x)', color='blue')
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

    x_vals, y_vals = tabulate(-2, 2, 0.1)
    coeffs = read_coeffs("coefficients.txt")
    print("1. Табуляцію завершено (див. tabulation.txt)")

    plot_graphs(x_vals, y_vals, coeffs)

    x_start = 0.7
    print("\n2. Трансцендентне рівняння (x^2 - exp(-x) = 0):")
    print(f"   Ньютон: {newton_method(x_start, EPS)}")
    print(f"   Чебишев: {chebyshev_method(x_start, EPS)}")
    print(f"   Метод Хорд: {secant_method(0.6, 0.8, EPS)}")

    print("\n3. Алгебраїчне рівняння (з coefficients.txt):")
    real_root, h_it = horner_newton(coeffs, 1.5, EPS)
    print(f"   Дійсний корінь (Схема Горнера): {real_root:.10f}, ітерацій: {h_it}")

    comp_root, l_it = lin_method(coeffs, 0.5, 0.5, EPS)
    print(f"   Комплексний корінь (Метод Ліна): {comp_root}, ітерацій: {l_it}")