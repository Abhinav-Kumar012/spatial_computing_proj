from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np

from train_eval import train_and_evaluate


class FusionOptimization(Problem):

    def __init__(self, train_path, test_path):

        super().__init__(
            n_var=5,
            n_obj=7,
            xl=np.array([16,16,4,0.01,1e-4]),
            xu=np.array([64,64,16,0.3,1e-3])
        )

        self.train_path=train_path
        self.test_path=test_path

    def _evaluate(self, X, out, *args, **kwargs):

        results=[]

        print(f"\n{'='*60}")
        print(f"[NSGA-2] Evaluating {len(X)} individual(s)")
        print(f"{'='*60}")

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
                "epochs":             5
            }

            print(f"\n[Individual {i+1}/{len(X)}]")
            print(f"  pan_ch={pan_target_channel} (raw={pan_raw})  "
                  f"ms_ch={ms_target_channel} (raw={ms_raw})  "
                  f"head_ch={head_channel}  "
                  f"dropout={params['dropout']:.3f}  lr={params['lr']:.2e}")

            metrics=train_and_evaluate(params,self.train_path,self.test_path)

            results.append([
                metrics["SAM"],
                metrics["ERGAS"],
                -metrics["CC"],
                metrics["SD"],
                -metrics["SF"],
                -metrics["SSIM"],
                metrics["ERGAS"]
            ])

            print(f"  -> objectives: SAM={metrics['SAM']:.4f}  ERGAS={metrics['ERGAS']:.4f}  "
                  f"CC={metrics['CC']:.4f}  SSIM={metrics['SSIM']:.4f}")

        out["F"]=np.array(results)


def run_nsga2(train_path,test_path):

    print("\n[run_nsga2] Starting NSGA-2 optimisation")
    print(f"  train: {train_path}")
    print(f"  test:  {test_path}")
    print(f"  pop_size=10  n_gen=10")

    problem=FusionOptimization(train_path,test_path)

    algorithm=NSGA2(pop_size=10)

    res=minimize(problem,
                 algorithm,
                 ('n_gen',10),
                 seed=1,
                 verbose=True)

    print("\n[run_nsga2] Optimisation complete")
    print(f"  Pareto front size: {len(res.F)}")

    return res