import numpy as np
import torch
from run_functions import run_AL
import pickle



# Parameters that can be changed
seed = 4200
n_train = 10000
n_query = 1000
n_rounds = 40
dataset = "CIFAR10"
strat = ["Random", "Uncertainty", "Margin"] #tells which strategy to use

with open("AL_scripts/results/parameters", "wb") as file:
    pickle.dump({"seed":seed, "n_train":n_train, "n_query":n_query, "n_rounds":n_rounds, "strategy":strat}, file)


if __name__ == "__main__":
    run_AL(n_train, n_query, n_rounds, dataset, strat, seed = seed)