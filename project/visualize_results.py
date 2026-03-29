import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def parallel_plot(results):
    # results is a numpy array (res.F); columns are objective indices
    df = pd.DataFrame(results)
    df.columns = [f"obj{i}" for i in range(df.shape[1])]

    # parallel_coordinates requires a class column — use the solution index
    df = df.reset_index()          # promotes integer index → column named "index"
    df["index"] = df["index"].astype(str)   # must be categorical/string

    plt.figure()
    parallel_coordinates(df, "index")
    plt.title("Pareto Front – Parallel Coordinates")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parallel_plot.png"), dpi=150)
    plt.show()


def pareto_2d(results, obj1, obj2):
    # results is a numpy array; obj1/obj2 are integer column indices
    plt.figure()
    plt.scatter(results[:, obj1], results[:, obj2])
    plt.xlabel(f"Objective {obj1}")
    plt.ylabel(f"Objective {obj2}")
    plt.title(f"Pareto Front (obj{obj1} vs obj{obj2})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"pareto_2d_obj{obj1}_obj{obj2}.png"), dpi=150)
    plt.show()


def pareto_3d(results, a, b, c):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(results[:, a], results[:, b], results[:, c])

    ax.set_xlabel(f"Objective {a}")
    ax.set_ylabel(f"Objective {b}")
    ax.set_zlabel(f"Objective {c}")
    plt.title(f"Pareto Front 3D (obj{a}, obj{b}, obj{c})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"pareto_3d_obj{a}_obj{b}_obj{c}.png"), dpi=150)
    plt.show()