from nsga2_search import run_nsga2
from visualize_results import *

data_folder = "../datav1"
train_path = f"{data_folder}/valid_wv3.h5"
test_path  = f"{data_folder}/test_wv3_OrigScale_multiExm1.h5"

# ── NSGA-2 search budget ───────────────────────────────────────────────────────
EPOCHS   = 5    # training epochs per candidate  (3–10 recommended)
POP_SIZE = 10   # NSGA-2 population size         (10–20 recommended)
N_GEN    = 10   # number of generations          (10–20 recommended)
# ──────────────────────────────────────────────────────────────────────────────

res = run_nsga2(train_path, test_path, epochs=EPOCHS, pop_size=POP_SIZE, n_gen=N_GEN)

results = res.F

parallel_plot(results)

pareto_2d(results, 0, 1)
pareto_2d(results, 0, 4)

pareto_3d(results, 0, 1, 5)