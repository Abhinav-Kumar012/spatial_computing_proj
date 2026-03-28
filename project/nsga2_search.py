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

        for x in X:

            head_channel = int(x[2])

            # ms_target_channel must be divisible by head_channel because
            # Attention splits it into num_head × head_channel for the reshape.
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

        out["F"]=np.array(results)


def run_nsga2(train_path,test_path):

    problem=FusionOptimization(train_path,test_path)

    algorithm=NSGA2(pop_size=10)

    res=minimize(problem,
                 algorithm,
                 ('n_gen',10),
                 seed=1,
                 verbose=True)

    return res