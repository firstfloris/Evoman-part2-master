sys.path.insert(0, 'evoman')
from operator import index
import sys
import os
from pyrsistent import v
from evoman_problem import Evoman
from regex import P, R
import pickle
from evoman.environment import Environment
from demo_controller import player_controller
import hulpfunctions as hf
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from crossover import TwoPointCrossover, NoCrossover
from mutation import SelfAdaptingMutation
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import plot
import multiprocessing
from dask.distributed import Client
from pymoo.core.problem import DaskParallelization
import dill
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP
import miscellaneous as misc
from tqdm import tqdm


################################################
if __name__ == '__main__':

    ################################################ 
    mode = "test"  
    ################################################

    enemies = [1,2,3,4,5,6,7,8]
    n_hidden = 10

    if mode == "test":
        repetitions = 1
        speed = "normal"
        sound = "on"
    else:
        repetitions = 5
        speed = "fastest"
        sound = "off"
        
    competition_name = f'comp_enemies_{enemies}_repetitions_{repetitions}_[{mode}]'
    hf.create_folder("competition")
    hf.create_folder(competition_name, add_date=True)
    