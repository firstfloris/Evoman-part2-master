from operator import index
from random import random, seed
import sys
sys.path.insert(0, 'evoman')
import os
from tabnanny import check, verbose
from sympy import comp
from evoman_problem import Evoman
from evoman.environment import Environment
from demo_controller import player_controller
import hulpfunctions as hf
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from crossover import NoCrossover, TwoPointSimpleRecombination
from mutation import SelfAdaptingMutation
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.termination.max_gen import MaximumGenerationTermination
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import plot
import warnings
import dill
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
import miscellaneous as misc
from tqdm import tqdm
sys.path.insert(0, 'evoman')
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------------------------------------------------------------------------------------

class MyCallback(Callback):

    def __init__(self, problem, pymoo_plot, run) -> None:
        super().__init__()
        self.data["best"] = []
        self.problem = problem
        self.pymoo_plot = pymoo_plot
        self.run = run

    def notify(self, algorithm):
        res = algorithm.pop.get("F")
        print(f"res: {res}")
        # if a file in f"{pymoo_plot}/scatter_{run}.pdf" that starts with "scatter_" exists, then delete the file
        if os.path.isfile(f"{self.pymoo_plot}/scatter_{self.run}.pdf"):
            os.remove(f"{self.pymoo_plot}/scatter_{self.run}.pdf")

        f_mean = self.problem.fitness_gen_mean
        print("f_mean", f_mean)
        f_min = self.problem.fitness_gen_min
        print("f_min", f_min)
        i = 0
        for e in enemies:
            plot.plot_fitness(experiment_name, len(f_mean[i]), f_mean[i], f_min[i], e, f"Enemy {e}", path=results_file, cp=True)
            i += 1

        Scatter(title="Pymoo NSGA2", axis_style={"xlim": [0, 1], "ylim": [0, 1]}).add(res).save(f"{self.pymoo_plot}/scatter_{self.run}.pdf")

def pymoo(print_ver, run):
    global envs

    envs = []
    for e in enemies:
        print(f"Creating environment for enemy group: {e}")
        envs.append(Environment(experiment_name='test',
                                multiplemode="yes",
                                enemies=e,
                                playermode="ai",
                                player_controller=player_controller(10),
                                enemymode="static",
                                level=2,
                                speed="fastest",
                                logs="off",
                                randomini="yes"))




    problem = Evoman(print_ver, envs, constants=constants)

    algorithm = NSGA2(pop_size=pop,
                    sampling=FloatRandomSampling(),
                    selection=RandomSelection(),
                    crossover=crossover_operator,
                    mutation=SelfAdaptingMutation(
                        min_std=min_std,
                        genome_length=constants['genome_length']),
                    survival=RankAndCrowdingSurvival(),
                    eliminate_duplicates=True,
                    n_offsprings=pop * 7)

    pymoo_plot= hf.create_folder(f"{results_file}/plots/pymoo")

    res = None

    if checkpoint_name == None:
        try:
            print("Starting optimization")
            res = minimize(problem, algorithm, ('n_gen', max_gen), verbose=True, callback=MyCallback(problem, pymoo_plot, run)) 
        except:
            print("Error in minimize, saving checkpoint")
            with open(f"{results_file}/checkpoint", "wb") as f:
                dill.dump(algorithm, f)
            print("Checkpoint saved")
            print("Exiting")
            raise

    else:
        try:
            with open(f"{results_file}/checkpoint", 'rb') as f:
                checkpoint = dill.load(f)
                print("Loaded Checkpoint:", checkpoint)
            checkpoint.termination = MaximumGenerationTermination(max_gen)

            res = minimize(problem, checkpoint, verbose=True, callback=MyCallback(problem, pymoo_plot, run))
        except:
            print("Error in minimizing checkpoint, saving checkpoint")
            with open(f"{results_file}/checkpoint", "wb") as f:
                dill.dump(algorithm, f)
            print("Checkpoint saved")
            print("Exiting")
            raise



    # if a file in f"{pymoo_plot}/scatter_{run}.pdf" that starts with "scatter_" exists, then delete the file
    if os.path.isfile(f"{pymoo_plot}/scatter_{run}.pdf"):
        os.remove(f"{pymoo_plot}/scatter_{run}.pdf")

    print("Plotting final results")
    Scatter(title="Pymoo NSGA2", axis_style={"xlim": [0, 1], "ylim": [0, 1]}).add(res.F).save(f"{pymoo_plot}/scatter_{run}.pdf")

    pr.nt(f'Threads: {round(res.exec_time, 2)}', verbose_level_of_text=0)
    # pr.nt(f"Fitness mean per generation: {round(problem.fitness_gen_mean, 3)}", verbose_level_of_text=1)
    # pr.nt(f"Fitness min per generation: {round(problem.fitness_gen_min, 3)}", verbose_level_of_text=1)

    return res, problem.fitness_gen_mean, problem.fitness_gen_min

################################################
if __name__ == '__main__':
    
    
    ######################################## PARAMETERS
    
    # set parameters
    max_gen = 5
    pop = 1
    inc_crossover_operator = True
    enemies = [[1,2,3,4],[5,6,7,8]]
    checkpoint_name = None
    competition=True
    test = True
    debug = False
    
    # pop = 50
    verbose_level = 0
    headless = True

    min_std = 0.01      #the min value for an std (mutation operator)

    ######################################## PARAMETERS


    constants = {
        'f_upper' : 100,
        'f_lower' : -10,
        'genome_length': 265
    }

    #start timer to time script
    start = dt.datetime.now()

    # crEate folder for results
    experiment_name = f'pymoo_algo_gen{max_gen}_pop{pop}_minstd{min_std}_crossover{inc_crossover_operator}_enemies{enemies[0]}{enemies[1]}'
    results_or_test = hf.define_parent_folder(test)
    if checkpoint_name == None or "None":
        results_file = hf.create_folder(f"{results_or_test}/{experiment_name}", add_date=True)
    else:
        results_file = f"{results_or_test}/{experiment_name}/{checkpoint_name}"
    plot_file = hf.create_folder(f"{results_file}/plots")

    #make print object to return progress updates
    pr = hf.print_verbose(accept_verbose_from=verbose_level)

    #determine which cross-over operator to use
    crossover_operator = TwoPointSimpleRecombination() if inc_crossover_operator else NoCrossover()

    #show games being played
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    if competition == False:
        N_runs = 10
    else:
        N_runs = 1

    result = {}
    # result[run][enemy][variable]
    for i in tqdm(range(1,N_runs+1), desc="Runs"):
        result[i] = {}
        res, f_gen_mean, f_gen_min = pymoo(pr, i)
        pr.nt(f"Result: {res}", verbose_level_of_text=0)

        for e in range(len(enemies)):
            result[i][e] = {}
            # result[i][e]['res'] = res[e]
            result[i][e]['f_gen_mean'] = f_gen_mean[e]
            result[i][e]['f_gen_min'] = f_gen_min[e]
            result[i][e]['f_mean'] = np.mean(f_gen_mean[e])
            result[i][e]['f_min'] = np.min(f_gen_min[e])


        pr.nt(f"F: {res.F}", verbose_level_of_text=4)
        # get the index of res.F where the values are closest to the [0,0] point, using euclidean distance
        # this is the index of the best solution
        index = np.argmin(np.linalg.norm(res.F, axis=1))
        
        best = res.X[index]
        result[i]['best_genome'] = best
        
        hf.save_model_as_object(result, f"{results_file}/result_dict_of_run{i}")

        pr.nt(f"Run {i} finished", verbose_level_of_text=5)

    # Plot fitness
    for e in range(len(enemies)):
        f_mean = []
        f_min = []
        for i in range(1,N_runs+1):
            f_mean.append(result[i][e]['f_gen_mean'])
            f_min.append(result[i][e]['f_gen_min'])
            pr.nt(f"Run {i} enemy {e} f_mean: {result[i][e]['f_mean']}", verbose_level_of_text=2)
            pr.nt(f"Run {i} enemy {e} f_min: {result[i][e]['f_min']}", verbose_level_of_text=2)

        pr.nt(f"Mean fitness for enemy {e}: {f_mean}", verbose_level_of_text=3)
        pr.nt(f"min fitness for enemy {e}: {f_min}", verbose_level_of_text=3)
        
        # if competition == True: 
        #     # plot.plot_fitness(experiment_name, len(f_mean[0]), f_mean[0], f_min[0], e, f"Enemy {e}", path=results_file, cp=True)
        # else: 
        #     plot.plot_fitness(experiment_name, len(f_mean), f_mean, f_min, e, f"Enemy {e}", path=results_file, cp=False)

    # Convert results to pd.DataFrame
    df = pd.DataFrame()
    for i in range(1,N_runs+1):
        for e in range(len(enemies)):
            df = df.append({'run': i, 'enemy': e, 'f_mean': result[i][e]['f_mean'], 'f_min': result[i][e]['f_min'], 'best_genome': result[i]['best_genome'], 'f_gen_mean': result[i][e]['f_gen_mean'], 'f_gen_min': result[i][e]['f_gen_min']}, ignore_index=True)

    # convert df to excel where each enemy is a sheet, and each run is a row in the sheet
    writer = pd.ExcelWriter(f'{results_file}/results_{N_runs}_runs.xlsx', engine='xlsxwriter')
    for e in range(len(enemies)):
        df[df['enemy'] == e].to_excel(writer, sheet_name=f'enemy_{e}')
    writer.save()

    # else:
    #     print("Competition mode")
    #     res, f_gen_mean, f_gen_min = pymoo(pr, 1)
    #     print(f"Training finished. Result saved.")

    
    # end timer
    end = dt.datetime.now()
    time_spend = end - start
    pr.nt(f"Time elapsed: {time_spend}", verbose_level_of_text=4)