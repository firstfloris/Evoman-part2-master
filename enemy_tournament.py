import datetime as dt
import hulpfunctions as hf
import miscellaneous as misc
from itertools import combinations
import pandas as pd
import sys, os
import numpy as np
import json

sys.path.insert(0, 'evoman')
from evoman_problem import Evoman
from evoman.environment import Environment
from demo_controller import player_controller
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from crossover import NoCrossover, TwoPointSimpleRecombination
from mutation import SelfAdaptingMutation
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
import plot


def pymoo(print_ver, run, combo):
    global envs

    envs = [] 
    for e in combo:
        envs.append(Environment(experiment_name='test',
                                enemies=[e],
                                playermode="ai",
                                player_controller=player_controller(10),
                                enemymode="static",
                                level=2,
                                speed="fastest",
                                logs="off",
                                randomini="yes"))

    # problem = Evoman(print_ver, envs, constants=constants, elementwise_runner=runner)
    problem = Evoman(print_ver, envs, constants=constants)

    algorithm = NSGA2(pop_size=pop,
                    sampling=FloatRandomSampling(),
                    selection=RandomSelection(),
                    crossover=TwoPointSimpleRecombination(),
                    mutation=SelfAdaptingMutation(
                        min_std=min_std,
                        genome_length=constants['genome_length']),
                    survival=RankAndCrowdingSurvival(),
                    eliminate_duplicates=True,
                    n_offsprings=pop * 7)


    if debug:
        res = minimize(problem,
            algorithm,
            ('n_gen', generations),
            verbose=True)
    else:
        # try expect block to catch ctrl+c or error and save results
        try:
            res = minimize(problem,
                    algorithm,
                    ('n_gen', generations),
                    verbose=True)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            misc.checkpointer(enemies, experiment_name, algorithm, problem, run, problem.out, path=results_file)
            raise

    pr.nt(f'Threads: {round(res.exec_time, 2)}', verbose_level_of_text=0)
    # pr.nt(f"Fitness mean per generation: {round(problem.fitness_gen_mean, 3)}", verbose_level_of_text=1)
    # pr.nt(f"Fitness min per generation: {round(problem.fitness_gen_min, 3)}", verbose_level_of_text=1)

    pymoo_plot= hf.create_folder(f"{results_file}/plots")

    Scatter(title="Pymoo NSGA2", axis_style={"xlim": [0, 1], "ylim": [0, 1]}).add(res.F).save(f"{pymoo_plot}/scatter_{run}.pdf")
    # PCP().add(res.F).save(f"{pymoo_plot}/pcp_{run}.pdf")
    # save res.F in csv
    df = pd.DataFrame(res.F)
    df.to_csv(f"{results_file}/pymoo_{run}.csv", index=False)

    return res, problem.fitness_gen_mean, problem.fitness_gen_min


if __name__ == '__main__':

    # set parameters
    test = True
    debug = False
    generations = 10
    pop = 5
    enemies = [1,2,3,4,5,6,7,8]
    verbose_level = 5
    headless = True

    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    min_std = 0.01      #the min value for an std (mutation operator)

    pr = hf.print_verbose(accept_verbose_from=verbose_level)

    constants = {
        'f_upper' : 100,
        'f_lower' : -10,
        'genome_length': 265
    }

    enemy_combos = [(1,2)]#list(combinations(enemies, 2))

    run = 0

    for combo in enemy_combos:
        run += 1

        experiment_name = f'{combo[0]}_and_{combo[1]}'
        results_file = hf.create_folder(f"combo_test/{experiment_name}", add_date=True)
        plot_file = hf.create_folder(f"combo_test/{experiment_name}/plots")

        result = {}

        res, f_gen_mean, f_gen_min = pymoo(pr, run, combo)

        population = np.split(res.pop.get("X"), 2, 1)

        genomes = population[0]

        for i in range(len(genomes)):
            result[i] = {}
            fitnesses = []
            defeated = []
            gain = []

            for e in enemies:
                env = Environment(experiment_name='test',
                                enemies=[e],
                                playermode="ai",
                                player_controller=player_controller(10),
                                enemymode="static",
                                level=2,
                                speed="fastest",
                                logs="off",
                                randomini="yes")

                f,p,e,t = env.play(pcont=genomes[i])
                print(f"fitness: {f}, player: {p}, enemy: {e}, time: {t}")
                fitnesses.append(f)
                gain.append(p - e)
                if e == 0: defeated.append(e)

            result[i]['genome'] = list(genomes[i])
            result[i]['combo'] = combo
            result[i]['fitnesses'] = fitnesses
            result[i]['defeated_enemies'] = defeated
            result[i]['gain'] = sum(gain)
            

            hf.save_model_as_object(result, f"{results_file}/results_of_run{i}")

    #Convert result dict to json
    with open(f'{results_file}/results_{N_runs}_runs.json', 'w') as convert_file:
        convert_file.write(json.dumps(result))
        # Convert results to pd.DataFrame


    df = pd.DataFrame()
    df.from_dict(result)

    # convert df to excel where each enemy is a sheet, and each run is a row in the sheet
    writer = pd.ExcelWriter(f'{results_file}/results_{N_runs}_runs.xlsx', engine='xlsxwriter')
    writer.save()