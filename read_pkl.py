import dill as pickle
import sys
import os
import numpy as np


def pretty_print(data, indent=0):
    prefix = " " * indent

    # ---- Dict ----
    if isinstance(data, dict):
        print(f"{prefix}{{")
        for key, value in data.items():
            print(f"{prefix}  {key}:")
            pretty_print(value, indent + 4)
        print(f"{prefix}}}")

    # ---- List / Tuple ----
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}[ (len={len(data)})")
        for i, item in enumerate(data):
            print(f"{prefix}  [{i}]:")
            pretty_print(item, indent + 4)
        print(f"{prefix}]")

    # ---- NumPy Array ----
    elif isinstance(data, np.ndarray):
        print(f"{prefix}NumPy Array:")
        print(f"{prefix}  shape: {data.shape}")
        print(f"{prefix}  dtype: {data.dtype}")

        # Print full array (be careful for large ones)
        np.set_printoptions(threshold=np.inf)
        print(f"{prefix}  values:\n{prefix}{data}")

    # ---- Everything else ----
    else:
        print(f"{prefix}{data} ({type(data)})")


def read_pkl(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found -> {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print("=== FULL PKL CONTENT ===\n")
        pretty_print(data)
        print("\n=== END ===")

    except Exception as e:
        print(f"Error reading PKL file: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_pkl.py <file.pkl>")
    else:
        read_pkl(sys.argv[1])