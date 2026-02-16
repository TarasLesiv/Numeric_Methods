import os

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_tabulation_latlon_elev(path, lat, lon, elev):
    with open(path, "w", encoding="utf-8") as f:
        f.write("i\tlat\tlon\telev_m\n")
        for i in range(len(elev)):
            f.write(f"{i}\t{lat[i]:.6f}\t{lon[i]:.6f}\t{elev[i]:.2f}\n")

def write_dist_elev(path, dist, elev):
    with open(path, "w", encoding="utf-8") as f:
        f.write("i\tdist_m\telev_m\n")
        for i in range(len(elev)):
            f.write(f"{i}\t{dist[i]:.2f}\t{elev[i]:.2f}\n")

def write_text(path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
