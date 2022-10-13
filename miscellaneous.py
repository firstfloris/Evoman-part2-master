import hulpfunctions as hf
import dill

def checkpointer(enemies, experiment_name, algorithm, problem, run, res, path):
    # create folder checkpoints
    hf.create_folder(f"{path}/checkpoints")

    # dump res
    hf.save_model_as_object(res, f"{path}/checkpoints/res_{run}")

    # dump algorithm
    hf.save_model_as_object(algorithm, f"{path}/checkpoints/algorithm_{run}")

    # # dump algorithm
    # with open(f"{path}/checkpoint_run_{run}", "wb") as f:
    #     dill.dump(algorithm, f)

    # save fitness of the run in a csv file
    with open(f"{path}/c_fitness_run_{run}.csv", "w") as f:
        f.write("gen,enemy,mean,max")
        for i in range(len(enemies)):
            for j in range(len(problem.fitness_gen_mean[i])):
                f.write(f"{j},{i},{problem.fitness_gen_mean[i][j]},{problem.fitness_gen_max[i][j]}")

def split_individual(X, genome_length):
    return X[:genome_length], X[genome_length:]
