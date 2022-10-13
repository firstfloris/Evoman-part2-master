from pymoo.core.problem import Problem
import numpy as np
import hulpfunctions as hf
from miscellaneous import split_individual

class Evoman(Problem):
    def __init__(self, print_ver, envs, constants, **kwargs):
        self.out = None
        self.gen_counter = 0
        self.envs = envs
        self.pr = print_ver
        self.constants = constants
        self.fitness_gen_mean = {}
        self.fitness_gen_min = {}
        self.genome_length = constants['genome_length']
        for i in range(len(envs)):
            self.fitness_gen_mean[i] = []
            self.fitness_gen_min[i] = []

        super().__init__(
            # n_var= 265, 
            n_var= self.genome_length * 2,
            n_obj=2,
            n_ieq_constr=0,
            xl=-1.0,
            xu=1.0,
            **kwargs)

    def _evaluate(self, genomes, out, *args, **kwargs):
        self.gen_counter += 1

        self.pr.nt(f"Generation {self.gen_counter}", verbose_level_of_text=2)
        res = []

        for x in genomes:
            res.append(self.eval(x=x, envs=self.envs))

        self.pr.nt(f"Fitness: {res}", verbose_level_of_text=3)

        # Make a f_g dictionary for each enemy
        f_g = {}
        for i in range(len(self.envs)):
            f_g[i] = []
            for j in range(len(res)):
                f_g[i].append(res[j][i])

        # Calculate the average and max fitness for each enemy
        for i in range(len(self.envs)):
            self.fitness_gen_mean[i].append(np.mean(f_g[i]))
            self.fitness_gen_min[i].append(np.min(f_g[i]))

        out["F"] = np.array(res)
        self.out = out


    def fitness_function(self, env):
        x = 0.8*(100 - env.get_enemylife()) + 0.2*env.get_playerlife() - np.log(env.get_time())
        norm_x = hf.normalize(x, max=self.constants['f_upper'], min=self.constants['f_lower'])

        return 1.0-norm_x

    def simulation_single(self, env,x,i):
        genome, _ = split_individual(x, self.genome_length)
        env.play(pcont=genome)
        f = self.fitness_function(env)
        p = env.get_playerlife()
        e = env.get_enemylife()
        t = env.get_time()
        self.pr.nt(text=f"(Enemy {i+1}) fitness: {f}, player: {p}, enemy: {e}, time: {t}", verbose_level_of_text=5)
        return f, p, e, t

    def eval(self, x, envs):
        res = []
        i = 0
        for env in envs:
            res.append(self.simulation_single(env,x[:self.genome_length],i)[0])
            i += 1
        self.pr.nt(f"fitness enemies: {res}", verbose_level_of_text=1, not_if_zero=True)
        return res