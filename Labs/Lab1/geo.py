import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """Відстань між двома GPS точками (метри)."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def cumulative_distance(lat, lon):
    """Кумулятивна дистанція вздовж маршруту (метри)."""
    n = len(lat)
    dist = np.zeros(n, dtype=float)
    for i in range(1, n):
        dist[i] = dist[i-1] + haversine(lat[i-1], lon[i-1], lat[i], lon[i])
    return dist
