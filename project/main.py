from nsga2_search import run_nsga2
from visualize_results import *
import dill
import os
import hashlib
import shutil

data_folder = "../datav1"
train_path = f"{data_folder}/valid_wv3.h5"
test_path  = f"{data_folder}/test_wv3_OrigScale_multiExm1.h5"

# ── NSGA-2 search budget ───────────────────────────────────────────────────────
EPOCHS   = 5    # training epochs per candidate  (3–10 recommended)
POP_SIZE = 25   # NSGA-2 population size         (Yields max 25 Pareto points)
N_GEN    = 20   # number of generations          (Provides enough search diversity)
# ──────────────────────────────────────────────────────────────────────────────

res = run_nsga2(train_path, test_path, epochs=EPOCHS, pop_size=POP_SIZE, n_gen=N_GEN)

# Save the final Pareto parameters (X) and objectives (F) explicitly mapped to each other
pareto_data = {
    "raw_X": res.X,
    "raw_F": res.F,
    "mapping": [{"decision": list(x), "result": list(f)} for x, f in zip(res.X, res.F)]
}

os.makedirs("results", exist_ok=True)
os.makedirs("results/pareto_models", exist_ok=True)

# Map the exact models from the temp folder based on their parameter signature
for item in pareto_data["mapping"]:
    x_hash = hashlib.md5(str(item["decision"]).encode()).hexdigest()
    model_name = f"model_{x_hash}.pt"
    
    source_model = os.path.join("results", "all_models", model_name)
    target_model = os.path.join("results", "pareto_models", model_name)
    
    # Explicity map as requested: "decision 1 : result 1 : model 1 name : model 1 path"
    item["model_name"] = model_name
    item["model_path"] = target_model
    
    if os.path.exists(source_model):
        shutil.move(source_model, target_model)

# Clean up temporary models created during NSGA-II to save huge amounts of disk space
if os.path.exists("results/all_models"):
    shutil.rmtree("results/all_models")

# Add top-level arrays for easier batch parsing during plotting
pareto_data["model_names"] = [item["model_name"] for item in pareto_data["mapping"]]
pareto_data["model_paths"] = [item["model_path"] for item in pareto_data["mapping"]]

# Overwrite the pickle file so that it contains the updated dictionary with paths integrated
with open("results/pareto_front.pkl", "wb") as f:
    dill.dump(pareto_data, f)
    
print("\n[main] Saved Pareto front parameters, objectives, and matched PyTorch weights to results/pareto_front.pkl")
print("Here is how your decisions directly map to the resulting metrics and model files:")

for i, mapped_item in enumerate(pareto_data["mapping"], start=1):
    print(f"  Decision {i}: {mapped_item['decision']}")
    print(f"  -> Result {i}: {mapped_item['result']}")
    print(f"  -> Model {i} Name: {mapped_item['model_name']}")
    print(f"  -> Model {i} Path: {mapped_item['model_path']}\n")
