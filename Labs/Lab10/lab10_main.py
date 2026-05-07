import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

def f(x, y):
    return x - y

def exact_solution(x):
    return x - 1 + 2 * np.exp(-x)

x0, y0 = 0.0, 1.0
a, b = 0.0, 5.0
h_initial = 0.01
epsilon = 1e-4


def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def print_results_table(method_name, x_vals, y_exact, y_approx, print_step=50):
    print(f"\n{'=' * 60}")
    print(f"Результати для: {method_name}")
    print(f"{'=' * 60}")
    print(f"{'x':<8} | {'Точне y(x)':<12} | {'Отримане y':<12} | {'Похибка':<15}")
    print("-" * 60)

    for i in range(0, len(x_vals), print_step):
        err = abs(y_exact[i] - y_approx[i])
        print(f"{x_vals[i]:<8.3f} | {y_exact[i]:<12.6f} | {y_approx[i]:<12.6f} | {err:<15.2e}")

    last_idx = len(x_vals) - 1
    if last_idx % print_step != 0:
        err = abs(y_exact[last_idx] - y_approx[last_idx])
        print(f"{x_vals[last_idx]:<8.3f} | {y_exact[last_idx]:<12.6f} | {y_approx[last_idx]:<12.6f} | {err:<15.2e}")
    print("-" * 60)


#Прогноз та корекція Адамса
def adams_fixed_step(a, b, y0, h):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    y_pred = np.zeros(len(x))
    y[0] = y0

    if len(x) > 1:
        y[1] = rk4_step(x[0], y[0], h)
        y_pred[1] = y[1]

    for i in range(1, len(x) - 1):
        y_p = y[i] + (h / 2) * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))
        y_pred[i + 1] = y_p
        y_c = y[i] + (h / 2) * (f(x[i + 1], y_p) + f(x[i], y[i]))
        y[i + 1] = y_c

    return x, y, y_pred


def adams_auto_step(a, b, y0, h_start, tol):
    x_vals = [a, a + h_start]
    y_vals = [y0, rk4_step(a, y0, h_start)]
    h_vals = [h_start, h_start]

    x_curr = x_vals[-1]
    h = h_start

    while x_curr < b:
        if x_curr + h > b:
            h = b - x_curr

        y_n, y_n_minus_1 = y_vals[-1], y_vals[-2]
        x_n, x_n_minus_1 = x_vals[-1], x_vals[-2]

        y_p = y_n + (h / 2) * (3 * f(x_n, y_n) - f(x_n_minus_1, y_n_minus_1))
        y_c = y_n + (h / 2) * (f(x_n + h, y_p) + f(x_n, y_n))

        error = abs(y_c - y_p)

        if error > tol:
            h /= 2
            x_vals.pop()
            y_vals.pop()
            h_vals.pop()
            x_curr = x_vals[-1]
            y_vals.append(rk4_step(x_curr, y_vals[-1], h))
            x_curr += h
            x_vals.append(x_curr)
            h_vals.append(h)
        else:
            x_curr += h
            x_vals.append(x_curr)
            y_vals.append(y_c)
            h_vals.append(h)
            if error < tol / 10:
                h *= 2

    return np.array(x_vals), np.array(y_vals), np.array(h_vals)


#Метод Рунге-Кутта 4 порядку
def rk4_fixed_step(a, b, y0, h):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(len(x) - 1):
        y[i + 1] = rk4_step(x[i], y[i], h)
    return x, y


def rk4_auto_step(a, b, y0, h_start, tol):
    x_vals = [a]
    y_vals = [y0]
    h_vals = [h_start]
    runge_errors = [0]

    x_curr, y_curr, h = a, y0, h_start

    while x_curr < b:
        if x_curr + h > b: h = b - x_curr

        y_full = rk4_step(x_curr, y_curr, h)
        y_half_1 = rk4_step(x_curr, y_curr, h / 2)
        y_half_2 = rk4_step(x_curr + h / 2, y_half_1, h / 2)

        err = (16 / 15) * abs(y_half_2 - y_full)

        if err > tol:
            h /= 2
        else:
            x_curr += h
            y_curr = y_full
            x_vals.append(x_curr)
            y_vals.append(y_curr)
            h_vals.append(h)
            runge_errors.append(err)
            if err < tol / 32: h *= 2

    return np.array(x_vals), np.array(y_vals), np.array(h_vals), np.array(runge_errors)

def main():

    x_adams, y_adams, y_pred = adams_fixed_step(a, b, y0, h_initial)
    y_exact_adams = exact_solution(x_adams)

    print_results_table("Метод Адамса (сталий крок h=0.01)", x_adams, y_exact_adams, y_adams, print_step=50)

    x_rk4, y_rk4 = rk4_fixed_step(a, b, y0, h_initial)
    y_exact_rk4 = exact_solution(x_rk4)

    print_results_table("Метод Рунге-Кутта 4-го порядку (сталий крок h=0.01)", x_rk4, y_exact_rk4, y_rk4, print_step=50)


    plt.figure()
    plt.plot(x_adams, np.abs(y_adams - y_exact_adams), label="|y_n - y(x_n)|")
    plt.title("Адамс: Локальна похибка")
    plt.xlabel("x")
    plt.ylabel("Похибка")
    plt.grid(True);
    plt.legend()
    plt.savefig(f"{output_dir}/adams_real_error.png")

    plt.figure()
    plt.plot(x_adams[1:], np.abs(y_adams - y_pred)[1:], label="|y_kor - y_pred|", color='orange')
    plt.title("Адамс: Оцінка похибки")
    plt.xlabel("x");
    plt.ylabel("Оцінка похибки")
    plt.grid(True);
    plt.legend()
    plt.savefig(f"{output_dir}/adams_est_error.png")

    x_ad_auto, y_ad_auto, h_ad_auto = adams_auto_step(a, b, y0, h_initial, epsilon)
    plt.figure()
    plt.step(x_ad_auto, h_ad_auto, where='post', color='green')
    plt.title("Адамс: Залежність величини кроку h(x)")
    plt.xlabel("x");
    plt.ylabel("Крок h")
    plt.grid(True)
    plt.savefig(f"{output_dir}/adams_auto_step.png")

    plt.figure()
    plt.plot(x_rk4, np.abs(y_rk4 - y_exact_rk4), label="|y_n - y(x_n)|", color='red')
    plt.title("РК4: Локальна похибка")
    plt.xlabel("x");
    plt.ylabel("Похибка")
    plt.grid(True);
    plt.legend()
    plt.savefig(f"{output_dir}/rk4_real_error.png")

    x_rk4_auto, y_rk4_auto, h_rk4_auto, err_runge = rk4_auto_step(a, b, y0, h_initial, epsilon)
    plt.figure()
    plt.plot(x_rk4_auto, err_runge, label="Похибка Рунге", color='purple')
    plt.title("РК4: Локальна похибка за правилом Рунге")
    plt.xlabel("x");
    plt.ylabel("Похибка")
    plt.grid(True);
    plt.legend()
    plt.savefig(f"{output_dir}/rk4_runge_error.png")

    plt.figure()
    plt.step(x_rk4_auto, h_rk4_auto, where='post', color='brown')
    plt.title("РК4: Залежність величини кроку h(x)")
    plt.xlabel("x");
    plt.ylabel("Крок h")
    plt.grid(True)
    plt.savefig(f"{output_dir}/rk4_auto_step.png")


if __name__ == "__main__":
    main()