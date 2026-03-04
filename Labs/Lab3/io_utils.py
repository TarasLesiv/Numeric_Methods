# io_utils.py
import csv
from pathlib import Path
from typing import List, Tuple


def read_csv_xy(path: str) -> Tuple[List[float], List[float]]:
    x, y = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "Month" not in reader.fieldnames or "Temp" not in reader.fieldnames:
            raise ValueError("CSV має містити заголовки Month,Temp")

        for row in reader:
            x.append(float(row["Month"]))
            y.append(float(row["Temp"]))

    if not x:
        raise ValueError("CSV порожній або не містить даних")
    return x, y


def ensure_default_csv(path: str) -> None:
   #Створення дефолту даних якщо нема файлу.
    p = Path(path)
    if p.exists():
        return

    default_rows = [
        (1, -2), (2, 0), (3, 5), (4, 10), (5, 15), (6, 20),
        (7, 23), (8, 22), (9, 17), (10, 10), (11, 5), (12, 0),
        (13, -10), (14, 3), (15, 7), (16, 13), (17, 19), (18, 20),
        (19, 22), (20, 21), (21, 18), (22, 15), (23, 10), (24, 3),
    ]

    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Month", "Temp"])
        for m, t in default_rows:
            w.writerow([m, t])

    print(f"[INFO] Файл {p.name} не знайдено — створення дефолтного {p.resolve()}")