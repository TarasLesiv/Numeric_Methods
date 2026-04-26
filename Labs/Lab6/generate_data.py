import random


def generate_lab_data(n=100, x_val=2.5):#генерація матриці з рандомом
    A = [[random.uniform(1, 100) + (1000 if i == j else 0) for j in range(n)] for i in range(n)]

    X_true = [x_val] * n

    B = []#обч B(вільні члени)
    for i in range(n):
        bi = sum(A[i][j] * X_true[j] for j in range(n))
        B.append(bi)

    with open('matrix_A.txt', 'w') as f:
        for row in A:
            f.write(' '.join(map(str, row)) + '\n')

    with open('vector_B.txt', 'w') as f:
        f.write(' '.join(map(str, B)) + '\n')

    print("Файли matrix_A.txt та vector_B.txt успішно створено.")


if __name__ == "__main__":
    generate_lab_data()