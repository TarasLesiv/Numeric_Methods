import matplotlib.pyplot as plt

def save_discrete_plot(path, dist, elev):
    plt.figure()
    plt.plot(dist, elev, marker="o")
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.title("Discrete elevation profile")
    plt.grid(True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def save_spline_plot(path, dist, elev, xx, yy, xk=None, yk=None, title="Spline"):
    plt.figure()
    plt.plot(dist, elev, "o", label="Data")
    plt.plot(xx, yy, "-", label="Spline")
    if xk is not None and yk is not None:
        plt.plot(xk, yk, "s", label="Chosen nodes")
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def save_error_plot(path, dist, err, title="Error"):
    plt.figure()
    plt.plot(dist, err, marker=".")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Distance (m)")
    plt.ylabel("Error (m)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
