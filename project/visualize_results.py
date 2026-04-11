import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

import dill
import itertools

# =============================================================================
# Variables for Visualization Configuration
# =============================================================================
# Pick which model results file to load (must be a .pkl containing Pareto data)
MODEL_DATA_PATH = os.path.join(os.path.dirname(__file__), "../resultsv4", "pareto_front.pkl")

# Define where to save the generated plots. 
# Creates directory at the level of the "project" folder (e.g., spatial_computing_proj/visualization_outputs)
project_root = os.path.dirname(os.path.dirname(__file__))
SAVE_DIR = os.path.join(project_root, "visualization_outputs")

os.makedirs(SAVE_DIR, exist_ok=True)
# =============================================================================

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

    # Normalize each column individually to [0, 1] to create an appropriate uniform scale
    for col in df.columns:
        if df[col].max() != df[col].min():
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # parallel_coordinates needs a string class column — use the solution index
    df = df.reset_index()
    df["index"] = df["index"].astype(str)

    plt.figure(figsize=(12, 5))
    parallel_coordinates(df, "index", colormap='viridis')
    plt.title("Pareto Front – Parallel Coordinates (Normalized [0,1])")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "parallel_plot.png"), dpi=150)
    plt.close()


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
    plt.savefig(os.path.join(SAVE_DIR, f"pareto_2d_{name1}_vs_{name2}.png"), dpi=150)
    plt.close()


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
    plt.savefig(os.path.join(SAVE_DIR, f"pareto_3d_{na}_{nb}_{nc}.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    print(f"Loading Pareto results from: {MODEL_DATA_PATH}")
    if not os.path.exists(MODEL_DATA_PATH):
        print("Error: The specified model data path does not exist.")
    else:
        with open(MODEL_DATA_PATH, "rb") as f:
            data = dill.load(f)
            
        # Extract the objectives from the parsed unpickled file
        if "raw_F" in data:
            results = data["raw_F"]
        elif "F" in data:
            results = data["F"]
        else:
            raise ValueError("Could not find objective data ('raw_F' or 'F') in pickle file.")
            
        print(f"Generating and saving plots to: {SAVE_DIR}")
        print("Creating PCP Plot...")
        parallel_plot(results)
        
        n_objs = results.shape[1]
        
        # All combinations of 2 objectives
        print("Generating all 2D combinations...")
        for obj1, obj2 in itertools.combinations(range(n_objs), 2):
            pareto_2d(results, obj1, obj2)
            
        # All combinations of 3 objectives
        print("Generating all 3D combinations...")
        for obj1, obj2, obj3 in itertools.combinations(range(n_objs), 3):
            pareto_3d(results, obj1, obj2, obj3)
            
        print("Done generating all possible Pareto frontier plots!")