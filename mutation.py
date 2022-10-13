import numpy as np

# from evoman_problem import population
# from Individual import Individual
from evoman_problem import Evoman
import miscellaneous as misc

from pymoo.core.mutation import Mutation
from pymoo.core.variable import get, Real
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside

# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------

class SelfAdaptingMutation(Mutation):

    def __init__(self, prob=0.9, eta=20, at_least_once=False, min_std:float=0.01, genome_length:int=265, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))
        self.genome_length = genome_length
        #min_std is the lower bound
        self.min_std = min_std
        #constants to determine learning rate
        self.tau_prime, self.tau = self.make_tau_and_prime()
        #track if mutation allready occured
        self.first_mutation = True

    def make_tau_and_prime(self):
        #create tau and tau' by formulas from the book
        tau = (1 / np.sqrt(2 * np.sqrt(self.genome_length)))
        tau_prime = (1 / np.sqrt(2 * self.genome_length))
        return tau_prime, tau

    def std_boundary(self, std:float)->float:
        # prevent std too close to zero
        if std < self.min_std:
            return self.min_std
        else:
            return std

    def mutate_std(self, current_std:float, individual_bias:float)->float:
        #mutate the standard deviations for every gen by a formula from the book
        new_std = current_std * np.exp(self.tau_prime * individual_bias + self.tau * np.random.normal(0,1))
        return self.std_boundary(new_std)

    def gen_boundaries(self, new_gen):
        #keep gen within the boundaries
        if new_gen < -1:
            return -1.0
        elif new_gen > 1:
            return 1.0
        else: 
            return new_gen            

    def mutate_gen(self, gen, std):
        #mutate gen with its corresponding standard deviation
        return self.gen_boundaries(gen + np.random.normal(0, std))

    def mutate_genome(self, genome, stds):
        #mutate all gens with its corresponding standard deviation
        genome_mut = np.array([self.mutate_gen(gen, stds[i]) for i, gen in enumerate(genome)])
        return genome_mut #array of genome

    def initiate_stds(self, stds):
        x = np.random.uniform(low=self.min_std, high=1,size=len(stds))
        return x

    def mutate_individual(self, x):
        #create a bias pas individual
        individual_bias = np.random.normal(0,1)

        #split the genome and standard deviations from the individual 
        genome, stds = misc.split_individual(x, self.genome_length)

        if self.first_mutation:
            #inititate stds properly because pymoo assigns random values as if they were weights/biases
            stds = self.initiate_stds(stds=stds)
            self.first_mutation = False

        #mutate the standard deviations
        stds = np.array([self.mutate_std(current_std=std, individual_bias=individual_bias) for std in stds])
        #mutate the genome with the standard deviations
        genome = np.array([self.mutate_genome(genome=genome, stds=stds)])

        #return the concatinations of the weights and biases and the standard deviations
        return np.append(genome, stds)


    def _do(self, problem:Evoman, X, params=None, **kwargs):
        X = X.astype(float)

        # mutate all individuals
        X_mut = np.array([self.mutate_individual(x=x) for x in X])

        #return the mutated population
        return X_mut

        eta = get(self.eta, size=len(X))
        prob_var = [(x[-1] - problem.xl[-1]) / (problem.xu[-1] - problem.xl[-1]) for x in X]
        
        Xp = mut_pm(X, problem.xl, problem.xu, eta, prob_var, at_least_once=self.at_least_once)
        return Xp

# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------


def mut_pm(X, xl, xu, eta, prob, at_least_once):
    n, n_var = X.shape
    assert len(eta) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = mut_binomial(n, n_var, prob, at_least_once=at_least_once)
    mut[:, xl == xu] = False

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]

    X = X[mut]
    eta = np.tile(eta[:, None], (1, n_var))[mut]

    delta1 = (X - _xl) / (_xu - _xl)
    delta2 = (_xu - X) / (_xu - _xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand = np.random.random(X.shape)
    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (_xu - _xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    # set the values for output
    Xp[mut] = _Y

    # in case out of bounds repair (very unlikely)
    Xp = set_to_bounds_if_outside(Xp, xl, xu)

    return Xp
