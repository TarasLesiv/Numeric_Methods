import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "lab5_assets"
os.makedirs(OUT_DIR, exist_ok=True)

def f(x):
    return 50 + 20*np.sin(np.pi * x / 12) + 5*np.exp(-0.2 * (x - 12)**2)

def exact_integral():
    return 1200 + 5 * math.sqrt(5 * math.pi) * math.erf(12 / math.sqrt(5))

def simpson_composite(N, a=0.0, b=24.0):
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule")
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

def adaptive_simpson(a=0.0, b=24.0, eps=1e-8, max_depth=50):
    cache = {}
    eval_count = 0

    def feval(x):
        nonlocal eval_count
        if x not in cache:
            cache[x] = 50 + 20 * math.sin(math.pi * x / 12) + 5 * math.exp(-0.2 * (x - 12)**2)
            eval_count += 1
        return cache[x]

    def simp(left, right, f_left, f_right, f_mid):
        return (right - left) / 6.0 * (f_left + 4.0 * f_mid + f_right)

    def recurse(left, right, f_left, f_right, f_mid, S, tol, depth):
        mid = (left + right) / 2.0
        q1 = (left + mid) / 2.0
        q3 = (mid + right) / 2.0

        f_q1 = feval(q1)
        f_q3 = feval(q3)

        S_left = simp(left, mid, f_left, f_mid, f_q1)
        S_right = simp(mid, right, f_mid, f_right, f_q3)

        if depth <= 0 or abs(S_left + S_right - S) <= 15.0 * tol:
            return S_left + S_right + (S_left + S_right - S) / 15.0

        return (
            recurse(left, mid, f_left, f_mid, f_q1, S_left, tol / 2.0, depth - 1)
            + recurse(mid, right, f_mid, f_right, f_q3, S_right, tol / 2.0, depth - 1)
        )

    f_a = feval(a)
    f_b = feval(b)
    mid = (a + b) / 2.0
    f_mid = feval(mid)
    S0 = simp(a, b, f_a, f_b, f_mid)
    value = recurse(a, b, f_a, f_b, f_mid, S0, eps, max_depth)
    return value, eval_count

def main():
    I0 = exact_integral()
    print(f"Exact integral I0 = {I0:.16f}")

    Ns = np.arange(10, 1001, 2)
    errors = np.array([abs(simpson_composite(int(N)) - I0) for N in Ns])
    first_ok = np.where(errors <= 1e-12)[0][0]
    Nopt = int(Ns[first_ok])
    epsopt = float(errors[first_ok])

    print(f"Nopt = {Nopt}")
    print(f"epsopt = {epsopt:.3e}")

    N0 = 8
    I2 = simpson_composite(2)
    I4 = simpson_composite(4)
    I8 = simpson_composite(8)

    eps0 = abs(I8 - I0)
    IR = I8 + (I8 - I4) / 15.0
    epsR = abs(IR - I0)
    p_est = math.log(abs((I4 - I2) / (I8 - I4)), 2.0)
    IA = I8 + (I8 - I4) / (2.0**p_est - 1.0)
    epsA = abs(IA - I0)

    print(f"I(8) = {I8:.16f}")
    print(f"eps0 = {eps0:.6e}")
    print(f"IR = {IR:.16f}, epsR = {epsR:.6e}")
    print(f"p_Aitken = {p_est:.6f}")
    print(f"IA = {IA:.16f}, epsA = {epsA:.6e}")

    eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    adaptive_rows = []
    for eps in eps_list:
        value, cnt = adaptive_simpson(eps=eps)
        adaptive_rows.append((eps, value, abs(value - I0), cnt))
    df_adaptive = pd.DataFrame(adaptive_rows, columns=["eps", "I_adapt", "abs_error", "eval_count"])
    df_adaptive.to_csv("adaptive_results.csv", index=False)

    # Plot 0.1
    x = np.linspace(0, 24, 1000)
    plt.figure(figsize=(8, 4.8))
    plt.plot(x, f(x))
    plt.title("Графік функції навантаження на сервер")
    plt.xlabel("Час x, год")
    plt.ylabel("Навантаження f(x)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "function_plot.png"), dpi=220)
    plt.close()

    # Plot 2
    plt.figure(figsize=(8, 4.8))
    plt.semilogy(Ns, errors, linewidth=1.8)
    plt.scatter([Nopt], [epsopt], s=28)
    plt.annotate(f"Nopt = {Nopt}\nε = {epsopt:.2e}", (Nopt, epsopt), textcoords="offset points", xytext=(8, 10), fontsize=9)
    plt.title("Похибка складової формули Сімпсона")
    plt.xlabel("Кількість розбиттів N")
    plt.ylabel("ε(N) = |I(N) - I0|")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "simpson_error_vs_N.png"), dpi=220)
    plt.close()

    # Plot 3
    eps_arr = df_adaptive["eps"].to_numpy(dtype=float)
    err_arr = np.maximum(df_adaptive["abs_error"].to_numpy(dtype=float), 1e-16)
    eval_arr = df_adaptive["eval_count"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 4.8))
    plt.loglog(eps_arr, err_arr, marker="o")
    plt.title("Адаптивний алгоритм: фактична похибка")
    plt.xlabel("Параметр ε")
    plt.ylabel("Фактична похибка |I_adapt - I0|")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "adaptive_error_vs_eps.png"), dpi=220)
    plt.close()

    # Plot 4
    plt.figure(figsize=(8, 4.8))
    plt.loglog(eps_arr, eval_arr, marker="o")
    plt.title("Адаптивний алгоритм: кількість обчислень f(x)")
    plt.xlabel("Параметр ε")
    plt.ylabel("Кількість обчислень функції")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "adaptive_evals_vs_eps.png"), dpi=220)
    plt.close()

    print("\nAdaptive results:")
    print(df_adaptive.to_string(index=False))

if __name__ == "__main__":
    main()
