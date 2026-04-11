# lab4_main.py
from pathlib import Path

from io_utils import read_csv_xy, ensure_default_csv
from lsq import best_fit, poly_values, error_table, forecast_next_months
from plots import plot_variances, plot_approximation, plot_error

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data.csv"


def main():
    ensure_default_csv(str(CSV_PATH))
    #читання даних з csv
    x, y = read_csv_xy(str(CSV_PATH))

    results, best = best_fit(x, y, 1, 4)

    print("Дисперсії для різних m:")
    for r in results:
        print(f"m={r.degree}: variance={r.var:.6f}")

    print(f"\nОптимальний степінь: m*={best.degree}, variance={best.var:.6f}")
    print(f"Коефіцієнти: {best.coef}")

    y_hat = poly_values(x, best.coef)

    x_future, y_future = forecast_next_months(x, best.coef, 3)
    print("\nПрогноз на наступні 3 місяці:")
    for xf, yf in zip(x_future, y_future):
        print(f"Month {int(xf)} -> {yf:.4f}")

    err = error_table(x, y, y_hat)

    plot_variances(results)
    plot_approximation(x, y, y_hat, best.degree)
    plot_error(err, best.degree)

    plt.show()


if __name__ == "__main__":
    main()