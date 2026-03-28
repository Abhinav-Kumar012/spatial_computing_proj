from nsga2_search import run_nsga2
from visualize_results import *

train_path="Dataset/WV3/train_wv3-001.h5"
test_path="Dataset/WV3/test_wv3_multiExm1.h5"

res=run_nsga2(train_path,test_path)

results=res.F

parallel_plot(results)

pareto_2d(results,0,1)
pareto_2d(results,0,4)

pareto_3d(results,0,1,5)