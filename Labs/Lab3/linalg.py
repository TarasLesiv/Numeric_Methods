from typing import List


def gauss_solve(A_in: List[List[float]], b_in: List[float]) -> List[float]:
    A = [row[:] for row in A_in]
    b = b_in[:]
    n = len(A)

    for k in range(n):
        max_row = k
        max_val = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > max_val:
                max_val = abs(A[i][k])
                max_row = i

        if max_val == 0.0:
            raise ValueError("Матриця вироджена або близька до виродженої (pivot=0).")

        if max_row != k:
            A[k], A[max_row] = A[max_row], A[k]
            b[k], b[max_row] = b[max_row], b[k]

        pivot = A[k][k]
        for i in range(k + 1, n):
            factor = A[i][k] / pivot
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += A[i][j] * x[j]
        if A[i][i] == 0.0:
            raise ValueError("Нуль на діагоналі під час зворотного ходу.")
        x[i] = (b[i] - s) / A[i][i]
    return x