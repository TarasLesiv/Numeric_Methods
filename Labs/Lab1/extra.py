import numpy as np

def total_length(dist):
    return float(dist[-1])

def ascent_descent(elev):
    de = np.diff(elev)
    ascent = float(np.sum(np.maximum(de, 0.0)))
    descent = float(np.sum(np.maximum(-de, 0.0)))
    return ascent, descent

def gradients_percent(dist, elev):
    """
    Повертає градієнти між сусідніми точками у %:
      grade_i = 100 * (Δelev / Δdist)
    """
    dd = np.diff(dist)
    de = np.diff(elev)
    # захист від ділення на 0
    grade = 100.0 * de / np.where(dd == 0, np.nan, dd)
    return grade  # довжина n-1

def segments_over_threshold(dist, grade, threshold=15.0):
    """
    Повертає список сегментів (start_m, end_m, length_m, max_grade_on_segment)
    де grade > threshold (тільки підйом).
    """
    mask = grade > threshold
    segs = []
    if not np.any(mask):
        return segs

    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        start_i = i
        max_g = grade[i]
        while i < len(mask) and mask[i]:
            max_g = max(max_g, grade[i])
            i += 1
        end_i = i  # segment ends at i
        start_m = float(dist[start_i])
        end_m = float(dist[end_i])
        segs.append((start_m, end_m, end_m - start_m, float(max_g)))
    return segs

def mechanical_work(mass_kg, total_ascent_m, g=9.81):
    """
    W = m * g * Δh
    Повертає (joules, kJ, kcal)
    """
    W = mass_kg * g * total_ascent_m
    kJ = W / 1000.0
    kcal = W / 4184.0
    return float(W), float(kJ), float(kcal)

def summary_text(dist, elev, mass_kg=80.0, steep_threshold=15.0):
    length_m = total_length(dist)
    ascent_m, descent_m = ascent_descent(elev)
    grade = gradients_percent(dist, elev)

    max_up = float(np.nanmax(grade))
    max_down = float(np.nanmin(grade))
    mean_abs = float(np.nanmean(np.abs(grade)))

    segs = segments_over_threshold(dist, grade, threshold=steep_threshold)

    W, kJ, kcal = mechanical_work(mass_kg, ascent_m)

    lines = []
    lines.append("=== Додаткові розрахунки ===")
    lines.append(f"Загальна довжина: {length_m:.2f} м ({length_m/1000:.3f} км)")
    lines.append(f"Сумарний набір висоти (ascent): {ascent_m:.2f} м")
    lines.append(f"Сумарний спуск (descent): {descent_m:.2f} м")
    lines.append("")
    lines.append("=== Градієнт (нахил) між сусідніми точками ===")
    lines.append(f"Макс підйом: {max_up:.2f} %")
    lines.append(f"Макс спуск: {max_down:.2f} %")
    lines.append(f"Середній |градієнт|: {mean_abs:.2f} %")
    lines.append("")
    lines.append(f"=== Ділянки з підйомом > {steep_threshold:.1f}% ===")
    if not segs:
        lines.append("Немає.")
    else:
        for j, (s, e, L, gmax) in enumerate(segs, 1):
            lines.append(f"{j}) {s:.1f}–{e:.1f} м (довжина {L:.1f} м), max {gmax:.2f}%")
    lines.append("")
    lines.append(f"=== Механічна робота підйому для {mass_kg:.0f} кг ===")
    lines.append(f"W = m*g*Δh = {W:.0f} Дж = {kJ:.2f} кДж = {kcal:.2f} ккал")

    return "\n".join(lines)
