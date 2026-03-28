import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

def parallel_plot(results):

    df=pd.DataFrame(results)

    plt.figure()

    parallel_coordinates(df,"index")

    plt.show()


def pareto_2d(results,obj1,obj2):

    plt.scatter(results[obj1],results[obj2])

    plt.xlabel(obj1)
    plt.ylabel(obj2)

    plt.show()


def pareto_3d(results,a,b,c):

    from mpl_toolkits.mplot3d import Axes3D

    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')

    ax.scatter(results[a],results[b],results[c])

    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_zlabel(c)

    plt.show()