import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask


def genomePointCrossover(X, n_points):
    # get the X of parents and count the matings
    _, n_matings, n_var = X.shape

    # start point of crossover
    r = np.row_stack([np.random.permutation(n_var - 1) + 1 for _ in range(n_matings)])[:, :n_points]
    r.sort(axis=1)
    r = np.column_stack([r, np.full(n_matings, n_var)])

    # the mask do to the crossover
    M = np.full((n_matings, n_var), False)

    # create for each individual the crossover range
    for i in range(n_matings):
        j = 0
        while j < r.shape[1] - 1:
            a, b = r[i, j], r[i, j + 1]
            M[i, a:b] = True
            j += 2

    Xp = crossover_mask(X, M)

    return Xp


def IntermediateRecombination(X, genome_length, a):
    # get the X of parents and count the matings
    _, n_matings, _ = X.shape

    for i in range(n_matings):
        k1 = np.random.randint(0, genome_length, 1)[0]
        k2 = np.random.randint(0, genome_length, 1)[0]

        X[0,i,k1:] = 0.5 * X[0,i,k1:] + 0.5 * X[1,i,k1:]
        X[1,i,k2:] = 0.5 * X[0,i,k2:] + 0.5 * X[1,i,k2:]

    return X


class PointCrossover(Crossover):
    def __init__(self, n_points, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points

    def _do(self, Problem, X, **kwargs):
        X = np.split(X, 2, 2)

        genomes, stds = X[0], X[1]

        genome_X = genomePointCrossover(genomes, self.n_points)
        stds_X = IntermediateRecombination(stds, Problem.genome_length, 0.5)


        return np.concatenate((genome_X, stds_X), 2)


class NoCrossover(Crossover):
    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, Problem, X, **kwargs):
        X = np.split(X, 2, 2)

        genomes, stds = X[0], X[1]

        stds_X = IntermediateRecombination(stds, Problem.genome_length, 0.5)

        return np.concatenate((genomes, stds_X), 2)


class TwoPointSimpleRecombination(PointCrossover):
    def __init__(self, **kwargs):
        super().__init__(n_points=2, **kwargs)