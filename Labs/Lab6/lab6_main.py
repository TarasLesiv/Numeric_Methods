import math


# --- Функції зчитування та запису ---

def read_matrix(filename):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            matrix.append([float(x) for x in line.split()])
    return matrix


def read_vector(filename):
    with open(filename, 'r') as f:
        line = f.read()
        return [float(x) for x in line.split()]


def save_lu(L, U, filename):
    n = len(L)
    with open(filename, 'w') as f:
        f.write("Matrix L:\n")
        for row in L:
            f.write(' '.join(map(str, row)) + '\n')
        f.write("\nMatrix U:\n")
        for row in U:
            f.write(' '.join(map(str, row)) + '\n')


# --- Алгоритми LU-розкладу ---

def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        U[i][i] = 1.0
    for k in range(n):# Обч рядка L
        for i in range(k, n):
            sum_l = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_l

        # Обч рядка U
        for i in range(k + 1, n):
            sum_u = sum(L[k][j] * U[j][i] for j in range(k))
            if L[k][k] == 0:
                raise ValueError("Ділення на нуль: Матриця вироджена або потребує вибору головного елемента.")
            U[k][i] = (A[k][i] - sum_u) / L[k][k]

    return L, U


def solve_lu(L, U, B):
    n = len(L)
    #прямий хід
    z = [0.0] * n
    z[0] = B[0] / L[0][0]
    for k in range(1, n):
        sum_z = sum(L[k][j] * z[j] for j in range(k))
        z[k] = (B[k] - sum_z) / L[k][k]

    #зворотній хід
    x = [0.0] * n
    x[n - 1] = z[n - 1]
    for k in range(n - 2, -1, -1):
        sum_x = sum(U[k][j] * x[j] for j in range(k + 1, n))
        x[k] = z[k] - sum_x
    return x

def matrix_vector_mult(A, x):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * x[j] for j in range(n))
    return res


def vector_norm(v):
    return max(abs(x) for x in v)

def main():
    eps_0 = 1e-14

    A = read_matrix('matrix_A.txt')
    B = read_vector('vector_B.txt')

    L, U = lu_decomposition(A)
    save_lu(L, U, 'matrix_LU.txt')

    X0 = solve_lu(L, U, B)

    Ax = matrix_vector_mult(A, X0)
    R = [B[i] - Ax[i] for i in range(len(B))]
    current_eps = vector_norm(R)
    print(f"Початкова точність (норма нев'язки): {current_eps:.2e}")

    X_current = X0
    iterations = 0

    while True:
        Ax = matrix_vector_mult(A, X_current)
        R = [B[i] - Ax[i] for i in range(len(B))]

        if vector_norm(R) <= eps_0 or iterations > 100:
            break

        deltaX = solve_lu(L, U, R)

        X_current = [X_current[i] + deltaX[i] for i in range(len(X_current))]
        iterations += 1

    print(f"Уточнена точність після {iterations} ітерацій: {vector_norm(R):.2e}")
    print(f"Перші 5 значень X: {X_current[:5]}")

if __name__ == "__main__":
    main()