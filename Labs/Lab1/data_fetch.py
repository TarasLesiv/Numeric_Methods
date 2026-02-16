import requests
import numpy as np

# URL з методички
URL = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

def fetch_route_points(url: str = URL):
    """Повертає numpy-масиви lat, lon, elev (м)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data["results"]

    lat = np.array([p["latitude"] for p in results], dtype=float)
    lon = np.array([p["longitude"] for p in results], dtype=float)
    elev = np.array([p["elevation"] for p in results], dtype=float)
    return lat, lon, elev
