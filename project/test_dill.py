import dill
from tqdm import tqdm

class MyClass:
    def __init__(self):
        self.pbar = tqdm(total=10)

obj = MyClass()
try:
    with open("test_dill.pkl", "wb") as f:
        dill.dump(obj, f)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
