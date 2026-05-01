import os
import dill
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Variables for Visualization Configuration
# =============================================================================
# Pick which model results file to load
MODEL_DATA_PATH = os.path.join(os.path.dirname(__file__), "../resultsv4", "pareto_front.pkl")

# Define where to save the generated plots
project_root = os.path.dirname(os.path.dirname(__file__))
SAVE_DIR = os.path.join(project_root, "visualization_outputs_pairwise_scatter_plot")

os.makedirs(SAVE_DIR, exist_ok=True)
# =============================================================================

# Objective names in the order returned by nsga2_search._evaluate.
OBJ_NAMES   = ["SAM", "ERGAS", "CC", "SD", "SF", "SSIM", "n_params"]
OBJ_NEGATED = {2, 4, 5}   # indices whose sign must be flipped back


def _to_display(results):
    """Return a copy of the results array with negated objectives flipped."""
    arr = np.array(results, dtype=float)
    for idx in OBJ_NEGATED:
        if idx < arr.shape[1]:
            arr[:, idx] *= -1
    return arr


def generate_pairwise_plot(results):
    arr = _to_display(results)
    df = pd.DataFrame(arr, columns=OBJ_NAMES[:arr.shape[1]])

    # Create pairwise plot with regression lines
    # Using corner=True is often helpful for large number of features
    # to avoid redundant plots, but we'll stick to the default to match the reference style.
    print("Creating pairwise plot with regression lines...")
    g = sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    
    g.fig.suptitle("Pareto Front - Pairwise Scatter Plot with Regression", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "pairwise_scatter_reg_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    
    plt.show()
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
            
        generate_pairwise_plot(results)
