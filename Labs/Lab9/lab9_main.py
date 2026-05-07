import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    #x1^2 + x2^2 - 4 = 0
    return x[0] ** 2 + x[1] ** 2 - 4

def f2(x):
    #x2 - exp(x1) = 0
    return x[1] - np.exp(x[0])

def objective_function(x):
    return f1(x) ** 2 + f2(x) ** 2

def exploratory_search(x_base, delta, q, eps1):
    x_current = np.copy(x_base)
    n = len(x_current)

    for i in range(n):
        f_old = objective_function(x_current)

        x_current[i] += delta[i]
        if objective_function(x_current) < f_old:
            continue

        x_current[i] -= 2 * delta[i]
        if objective_function(x_current) < f_old:
            continue

        x_current[i] += delta[i]

    return x_current

def hooke_jeeves(x0, delta, eps1, eps2, q=2.0, p=2.0):
    x_prev_base = np.copy(x0)
    d = np.copy(delta)
    trajectory = [x_prev_base.copy()]

    x_curr_base = exploratory_search(x_prev_base, d, q, eps1)

    while True:
        if not np.array_equal(x_curr_base, x_prev_base):
            if np.linalg.norm(x_curr_base - x_prev_base) < eps1 or \
                    abs(objective_function(x_curr_base) - objective_function(x_prev_base)) < eps2:
                return x_curr_base, trajectory

            x_pattern = x_curr_base + p * (x_curr_base - x_prev_base)
            x_after_pattern = exploratory_search(x_pattern, d, q, eps1)

            if objective_function(x_after_pattern) < objective_function(x_curr_base):
                x_prev_base = x_curr_base
                x_curr_base = x_after_pattern
                trajectory.append(x_curr_base.copy())
            else:
                x_prev_base = x_curr_base
        else:
            if np.max(d) < eps1:
                return x_prev_base, trajectory
            d /= q
            x_curr_base = exploratory_search(x_prev_base, d, q, eps1)


if __name__ == "__main__":
    X0 = np.array([-1.0, 1.0])
    DELTA = np.array([0.2, 0.2])  #Початковий крок
    EPS1 = 1e-4  #Точність по координатах
    EPS2 = 1e-5  #Точність по функції
    Q_STEP = 2.0  #Коефіцієнт зменшення кроку
    P_ACCEL = 1.0  #Коефіцієнт прискорення p

    print("Пошук мінімуму методом Хука-Дживса:")
    result, path = hooke_jeeves(X0, DELTA, EPS1, EPS2, Q_STEP, P_ACCEL)

    with open("trajectory.txt", "w", encoding="utf-8") as f:
        f.write("Крок\tX1\t\tX2\t\tPhi(X)\n")
        for i, pt in enumerate(path):
            f.write(f"{i}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{objective_function(pt):.6e}\n")

    print(f"Результат: x* = {result}")
    print(f"Значення функції: {objective_function(result)}")
    print(f"Кількість кроків траєкторії: {len(path)}")
    print("Дані збережено у 'trajectory.txt'")


    x_range = np.linspace(-3, 3, 400)
    y_range = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X ** 2 + Y ** 2 - 4) ** 2 + (Y - np.exp(X)) ** 2

    plt.figure(figsize=(10, 8))

    cp = plt.contour(X, Y, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
    plt.clabel(cp, inline=True, fontsize=8)

    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'r-o', markersize=4, linewidth=1, label='Траєкторія')
    plt.plot(X0[0], X0[1], 'go', label='Старт')
    plt.plot(result[0], result[1], 'bo', label='Фініш (мінімум)')

    plt.title('Траєкторія спуску методом Хука-Дживса')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig('optimization_plot.png', dpi=300)
    print("Графік збережено у 'optimization_plot.png'")

    plt.show()
