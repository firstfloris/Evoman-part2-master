# plot the results with matplotlib with the results from neat_algo_V1.py

from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import mean_gamma_deviance
import hulpfunctions as hf

def plot_fitness(experiment_name, generations, fitness_mean, fitness_min, enemy_type, title, path='', cp=False):
    
    if cp == False:
        mean_gens = np.mean(fitness_mean, axis=0)
        std_gens = np.std(fitness_mean, axis=0)
        
        mean_min = np.mean(fitness_min, axis=0)
        fit_min = np.min(fitness_min, axis=0)
        std_min = np.std(fitness_min, axis=0)
    else:
        mean_gens = fitness_mean
        std_gens = np.std(fitness_mean)
        
        mean_min = fitness_min
        fit_min = np.min(fitness_min)
        std_min = np.std(fitness_min)
    
    x = np.arange(len(mean_gens))
    plt.xlabel("Generation", fontsize=18)
    plt.ylabel("Fitness", fontsize=18)
    plt.suptitle(f"Fitness per generation for enemy {enemy_type}", fontsize=18)
    
    plt.title(title)
    plt.plot(x, mean_gens, label="mean", linestyle = "solid")
    plt.fill_between(x, mean_gens - std_gens, mean_gens + std_gens, alpha=0.2)
    plt.plot(x, mean_min, label="min", linestyle = "solid")
    plt.fill_between(x, mean_min - std_min, mean_min + std_min, alpha=0.2)
    plt.legend()
    
    # plt.plot(x, mean_gens, label="mean", linestyle = "solid")
    # plt.fill_between(np.arange(0, len(mean_gens)), mean_gens - std_gens, mean_gens + std_gens, alpha=0.2)
    # plt.plot(x, mean_min, label="min", linestyle = "solid")
    # plt.fill_between(np.arange(0, len(fit_min)), fit_min - std_min, fit_min + std_min, alpha=0.2, hatch='/')
    # plt.legend()
    
    # create plots folder if it doesn't exist
    hf.create_folder(f"{path}/plots")
    plt.savefig(f"{path}/plots/fitness_enemy{enemy_type}.pdf")
    plt.clf()