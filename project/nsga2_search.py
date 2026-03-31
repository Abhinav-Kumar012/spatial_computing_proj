from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
import numpy as np
from tqdm import tqdm
import dill
import os

from train_eval import train_and_evaluate

class CheckpointCallback(Callback):
    def __init__(self, filepath="nsga2_checkpoint.pkl"):
        super().__init__()
        self.filepath = filepath
        self.tmp_filepath = filepath + ".tmp"

    def notify(self, algorithm):
        # Handle the tqdm pbar before pickling to avoid serialization errors
        pbar = getattr(algorithm.problem, "pbar", None)
        if pbar is not None:
            algorithm.problem.pbar = None
            
        with open(self.tmp_filepath, "wb") as f:
            dill.dump(algorithm, f)
        os.replace(self.tmp_filepath, self.filepath)
        
        # Restore the pbar reference
        if pbar is not None:
            algorithm.problem.pbar = pbar


class FusionOptimization(Problem):

    def __init__(self, train_path, test_path, epochs=5, pop_size=10, n_gen=10):

        super().__init__(
            n_var=5,
            n_obj=7,
            xl=np.array([16,16,4,0.01,1e-4]),
            xu=np.array([64,64,16,0.3,1e-3])
        )

        self.train_path = train_path
        self.test_path  = test_path
        self.epochs     = epochs
        self.pop_size   = pop_size
        self.n_gen      = n_gen
        self.pbar       = tqdm(total=self.pop_size * self.n_gen, desc="Optimization Process", unit="model")

    def _evaluate(self, X, out, *args, **kwargs):

        results=[]

        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"[NSGA-2] Evaluating {len(X)} individual(s)")
        tqdm.write(f"{'='*60}")

        for i, x in enumerate(X):

            head_channel = int(x[2])

            # ms_target_channel must be divisible by head_channel because
            # Attention splits it into num_head x head_channel for the reshape.
            # Round down to the nearest valid multiple (minimum = head_channel).
            ms_raw = int(x[1])
            ms_target_channel = max(head_channel, (ms_raw // head_channel) * head_channel)

            # Apply the same rounding to pan_target_channel for consistency.
            pan_raw = int(x[0])
            pan_target_channel = max(head_channel, (pan_raw // head_channel) * head_channel)

            params={
                "pan_target_channel": pan_target_channel,
                "ms_target_channel":  ms_target_channel,
                "head_channel":       head_channel,
                "dropout":            float(x[3]),
                "lr":                 float(x[4]),
                "epochs":             self.epochs
            }

            tqdm.write(f"\n[Individual {i+1}/{len(X)}]")
            tqdm.write(f"  pan_ch={pan_target_channel} (raw={pan_raw})  "
                  f"ms_ch={ms_target_channel} (raw={ms_raw})  "
                  f"head_ch={head_channel}  "
                  f"dropout={params['dropout']:.3f}  lr={params['lr']:.2e}")

            metrics=train_and_evaluate(params,self.train_path,self.test_path)

            results.append([
                metrics["SAM"],        # obj 0 — minimise
                metrics["ERGAS"],      # obj 1 — minimise
                -metrics["CC"],        # obj 2 — minimise (↑ CC)
                metrics["SD"],         # obj 3 — minimise
                -metrics["SF"],        # obj 4 — minimise (↑ SF)
                -metrics["SSIM"],      # obj 5 — minimise (↑ SSIM)
                metrics["n_params"],   # obj 6 — minimise (smaller model)
            ])

            tqdm.write(f"  -> objectives: SAM={metrics['SAM']:.4f}  ERGAS={metrics['ERGAS']:.4f}  "
                  f"CC={metrics['CC']:.4f}  SSIM={metrics['SSIM']:.4f}  "
                  f"n_params={metrics['n_params']:,}")
            
            self.pbar.update(1)

        out["F"]=np.array(results)


def run_nsga2(train_path, test_path, epochs=5, pop_size=10, n_gen=10):

    print("\n[run_nsga2] Starting NSGA-2 optimisation")
    print(f"  train: {train_path}")
    print(f"  test:  {test_path}")
    print(f"  epochs={epochs}  pop_size={pop_size}  n_gen={n_gen}")

    checkpoint_file = "nsga2_checkpoint.pkl"

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            algorithm = dill.load(f)
            print(f"  -> Resuming from checkpoint: generation {algorithm.n_gen}")
            
        problem = algorithm.problem
        # Calculate how many evaluations have occurred and resume the progress bar
        evaluated = algorithm.n_gen * problem.pop_size
        problem.pbar = tqdm(total=problem.pop_size * problem.n_gen, initial=evaluated, desc="Optimization Process (Resumed)", unit="model")
    else:
        problem  = FusionOptimization(train_path, test_path, epochs=epochs, pop_size=pop_size, n_gen=n_gen)
        algorithm = NSGA2(pop_size=pop_size)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   seed=1,
                   verbose=False,
                   copy_algorithm=False,
                   callback=CheckpointCallback(checkpoint_file))
                   
    problem.pbar.close()

    print("\n[run_nsga2] Optimisation complete")
    print(f"  Pareto front size: {len(res.F)}")

    return res