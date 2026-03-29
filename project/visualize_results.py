import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Objective names in the order returned by nsga2_search._evaluate.
# Negated objectives (CC, SF, SSIM) are stored as negative values by NSGA-2
# so we flip them back here for human-readable plots.
OBJ_NAMES   = ["SAM", "ERGAS", "CC", "SD", "SF", "SSIM", "n_params"]
OBJ_NEGATED = {2, 4, 5}   # indices whose sign must be flipped back


def _to_display(results):
    """Return a copy of the results array with negated objectives flipped."""
    arr = np.array(results, dtype=float)
    for idx in OBJ_NEGATED:
        if idx < arr.shape[1]:
            arr[:, idx] *= -1
    return arr


def parallel_plot(results):
    arr = _to_display(results)
    df  = pd.DataFrame(arr, columns=OBJ_NAMES[: arr.shape[1]])

    # parallel_coordinates needs a string class column — use the solution index
    df = df.reset_index()
    df["index"] = df["index"].astype(str)

    plt.figure(figsize=(12, 5))
    parallel_coordinates(df, "index")
    plt.title("Pareto Front – Parallel Coordinates")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parallel_plot.png"), dpi=150)
    plt.show()


def pareto_2d(results, obj1, obj2):
    arr   = _to_display(results)
    name1 = OBJ_NAMES[obj1]
    name2 = OBJ_NAMES[obj2]

    plt.figure()
    plt.scatter(arr[:, obj1], arr[:, obj2])
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(f"Pareto Front — {name1} vs {name2}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"pareto_2d_{name1}_vs_{name2}.png"), dpi=150)
    plt.show()


def pareto_3d(results, a, b, c):
    from mpl_toolkits.mplot3d import Axes3D

    arr   = _to_display(results)
    na, nb, nc = OBJ_NAMES[a], OBJ_NAMES[b], OBJ_NAMES[c]

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(arr[:, a], arr[:, b], arr[:, c])
    ax.set_xlabel(na)
    ax.set_ylabel(nb)
    ax.set_zlabel(nc)
    plt.title(f"Pareto Front 3D — {na}, {nb}, {nc}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"pareto_3d_{na}_{nb}_{nc}.png"), dpi=150)
    plt.show()