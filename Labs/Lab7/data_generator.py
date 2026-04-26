import numpy as np


def generate_data(n=100, filename_a="matrix_a.txt", filename_b="vector_b.txt"):
    A = np.random.uniform(-10, 10, (n, n))
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = 1 * (row_sum + np.random.uniform(1, 10))

    x_exact = np.full(n, 2.5)
    b = A @ x_exact

    np.savetxt(filename_a, A)
    np.savetxt(filename_b, b)
    print(f"Дані згенеровано: {filename_a}, {filename_b}")


if __name__ == "__main__":
    generate_data()