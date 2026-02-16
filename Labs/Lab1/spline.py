import numpy as np

def thomas(a, b, c, d):
    """
    Метод прогонки для тридіагональної СЛАР:
      a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    де a[0]=0, c[-1]=0
    Повертає x та (cp, dp) — коефіцієнти прогонки для звіту.
    """
    m = len(d)
    cp = np.zeros(m, dtype=float)
    dp = np.zeros(m, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, m):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = (c[i] / denom) if i < m - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.zeros(m, dtype=float)
    x[-1] = dp[-1]
    for i in range(m - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x, cp, dp

def natural_cubic_spline_coeffs(x, y):
    """
    Natural cubic spline: S''(x0)=0, S''(xn)=0.
    Повертає (ai, bi, ci, di, debug_dict)
    де debug_dict має дані для виводу в консоль/файл.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("Потрібно >=3 вузлів")

    h = np.diff(x)
    m = n - 2  # внутрішні вузли

    a = np.zeros(m, dtype=float)
    b = np.zeros(m, dtype=float)
    c = np.zeros(m, dtype=float)
    d = np.zeros(m, dtype=float)

    for k in range(1, n - 1):  # k=1..n-2
        i = k - 1
        hkm1 = x[k] - x[k - 1]
        hk   = x[k + 1] - x[k]
        a[i] = hkm1
        b[i] = 2 * (hkm1 + hk)
        c[i] = hk
        d[i] = 6 * ((y[k + 1] - y[k]) / hk - (y[k] - y[k - 1]) / hkm1)

    a[0] = 0.0
    c[-1] = 0.0

    M_inner, cp, dp = thomas(a, b, c, d)

    M = np.zeros(n, dtype=float)
    M[1:-1] = M_inner  # M0=Mn-1=0

    ai = y[:-1]
    bi = (y[1:] - y[:-1]) / h - (h * (2 * M[:-1] + M[1:])) / 6
    ci = M[:-1] / 2
    di = (M[1:] - M[:-1]) / (6 * h)

    debug = {
        "a_tridiag": a, "b_tridiag": b, "c_tridiag": c, "d_rhs": d,
        "cp": cp, "dp": dp, "M_inner": M_inner,
        "ai": ai, "bi": bi, "ci": ci, "di": di
    }
    return ai, bi, ci, di, debug

def spline_eval(x_nodes, ai, bi, ci, di, xq):
    """Обчислює значення сплайна в точках xq."""
    x_nodes = np.asarray(x_nodes, dtype=float)
    xq = np.asarray(xq, dtype=float)

    idx = np.searchsorted(x_nodes, xq) - 1
    idx = np.clip(idx, 0, len(x_nodes) - 2)

    dx = xq - x_nodes[idx]
    return ai[idx] + bi[idx]*dx + ci[idx]*dx**2 + di[idx]*dx**3
