import numpy as np

from data_fetch import fetch_route_points
from geo import cumulative_distance
from spline import natural_cubic_spline_coeffs, spline_eval
from io_utils import ensure_dir, write_tabulation_latlon_elev, write_dist_elev, write_text
from plots import save_discrete_plot, save_spline_plot, save_error_plot
from extra import summary_text



def pick_nodes(x, y, k):
    idx = np.linspace(0, len(x) - 1, k).round().astype(int)
    idx = np.unique(idx)
    return x[idx], y[idx]


def format_debug(debug):
    # Робимо текст, який піде у console_log.txt (коефіцієнти прогонки, розв'язок, коеф. сплайна)
    lines = []
    lines.append("[Тридіагональна СЛАР]")
    lines.append(f"a = {debug['a_tridiag']}")
    lines.append(f"b = {debug['b_tridiag']}")
    lines.append(f"c = {debug['c_tridiag']}")
    lines.append(f"d = {debug['d_rhs']}")
    lines.append("")
    lines.append("[Прогонка]")
    lines.append(f"c' = {debug['cp']}")
    lines.append(f"d' = {debug['dp']}")
    lines.append(f"x  = {debug['M_inner']}  (розв'язок для внутрішніх M)")
    lines.append("")
    lines.append("[Коефіцієнти сплайнів на інтервалах]")
    lines.append(f"ai = {debug['ai']}")
    lines.append(f"bi = {debug['bi']}")
    lines.append(f"ci = {debug['ci']}")
    lines.append(f"di = {debug['di']}")
    lines.append("")
    return "\n".join(lines)


def main():
    # КУДИ ЗАПИСУЄМО ВСЕ:
    out_dir = "outputs"
    plots_dir = "outputs/plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    # 1) API -> lat/lon/elev
    lat, lon, elev = fetch_route_points()

    # 2) Запис табуляції (обов'язковий файл)
    write_tabulation_latlon_elev(f"{out_dir}/tabulation.txt", lat, lon, elev)

    # 3) Кумулятивна дистанція + запис dist/elev
    dist = cumulative_distance(lat, lon)
    write_dist_elev(f"{out_dir}/dist_elev.txt", dist, elev)

    # 4) Дискретний графік
    save_discrete_plot(f"{plots_dir}/discrete.png", dist, elev)

    # 5) Сплайн на всіх вузлах
    ai, bi, ci, di, debug = natural_cubic_spline_coeffs(dist, elev)
    xx = np.linspace(dist[0], dist[-1], 1000)
    yy = spline_eval(dist, ai, bi, ci, di, xx)
    save_spline_plot(f"{plots_dir}/spline_all.png", dist, elev, xx, yy, title="Spline (all nodes)")

    # Логи (прогонка + коефіцієнти) для all-nodes
    log_text = ["=== ALL NODES ===", format_debug(debug)]

    # 6) 10/15/20 вузлів + похибка + метрики + графіки
    for k in (10, 15, 20):
        xk, yk = pick_nodes(dist, elev, k)
        a2, b2, c2, d2, dbg2 = natural_cubic_spline_coeffs(xk, yk)

        # Прогноз на всіх оригінальних точках dist
        y_pred = spline_eval(xk, a2, b2, c2, d2, dist)
        err = y_pred - elev
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))

        log_text.append(f"\n=== k={k} ===")
        log_text.append(f"RMSE={rmse} m, MAE={mae} m")
        log_text.append(format_debug(dbg2))

        # Графіки
        xxk = np.linspace(dist[0], dist[-1], 1000)
        yyk = spline_eval(xk, a2, b2, c2, d2, xxk)

        save_spline_plot(
            f"{plots_dir}/spline_{k}.png",
            dist, elev, xxk, yyk,
            xk=xk, yk=yk,
            title=f"Spline (k={k})"
        )
        save_error_plot(
            f"{plots_dir}/err_{k}.png",
            dist, err,
            title=f"Error (k={k})"
        )
        extra_txt = summary_text(dist, elev, mass_kg=80.0, steep_threshold=15.0)
        write_text(f"{out_dir}/extra.txt", extra_txt)

    # 7) Запис логів в outputs/console_log.txt
    write_text(f"{out_dir}/console_log.txt", "\n".join(log_text))

if __name__ == "__main__":
    main()
