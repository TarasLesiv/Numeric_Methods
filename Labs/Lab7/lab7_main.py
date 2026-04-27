import numpy as np


def load_data():
    return np.loadtxt("matrix_a.txt"), np.loadtxt("vector_b.txt")


def vector_norm(v):
    return np.max(np.abs(v))


def matrix_norm(A):
    return np.max(np.sum(np.abs(A), axis=1))


def simple_iteration(A, b, eps, max_iter=10000):
    n = len(b)
    tau = 1.0 / matrix_norm(A)
    x = np.ones(n)
    for k in range(max_iter):
        residual = A @ x - b
        x_new = x - tau * residual
        diff = vector_norm(x_new - x)
        if diff > 1e10:
            print("Метод простої ітерації розбігається!")
            return x, k
        if diff < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter


def jacobi_method(A, b, eps, max_iter=10000):
    n = len(b)
    x = np.ones(n)
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i, i]
        if vector_norm(x_new - x) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter


def gauss_zeidel_method(A, b, eps, max_iter=10000):
    n = len(b)
    x = np.ones(n)
    for k in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]
        if vector_norm(x - x_prev) < eps:
            return x, k + 1
    return x, max_iter


if __name__ == "__main__":
    try:
        A, b = load_data()
        eps = 1e-14

        for name, func in [("Проста ітерація", simple_iteration),
                           ("Якобі", jacobi_method),
                           ("Зейдель", gauss_zeidel_method)]:
            sol, iters = func(A, b, eps)
            err = vector_norm(sol - 2.5)
            print(f"{name:15} | Ітерацій: {iters:4} | Похибка: {err:.2e}")
    except FileNotFoundError:
        print("Помилка: Спочатку запустити data_generator.py")