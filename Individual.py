import numpy as np

class Individual():
    def __init__(self, x_vector, length_genome:int=265) -> None:
        #constant determined at initiation
        self.length_genome = length_genome
        #weights and biases
        self.genome = x_vector[:length_genome]
        #self adaptive mutation power (standard devations)
        self.stds = x_vector[length_genome:]
        #std of the individual (a bias for every individual set at the mutation)
        self.genome_std = None

    def reset_x(self):
        return np.concatenate((self.genome, self.stds, self.genome_std))

    # def init_strat_params(self, )