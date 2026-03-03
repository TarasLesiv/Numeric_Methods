from __future__ import annotations
from typing import List, Tuple
import csv


def read_csv_two_columns(filename: str) -> Tuple[List[float], List[float]]:
    """
    Зчитує CSV з 2 числовими колонками (будь-які назви).
    Повертає x, y як списки float.
    """
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Порожній CSV")
        x: List[float] = []
        y: List[float] = []
        for row in reader:
            if not row or all((c.strip() == "" for c in row)):
                continue
            if len(row) < 2:
                raise ValueError("У кожному рядку має бути принаймні 2 значення")
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y


def write_tabulation(filename: str, x: List[float], y: List[float], colx: str = "x", coly: str = "y") -> None:
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([colx, coly])
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])
