import numpy as np
import torch
from run_functions import run_AL
import pickle



# Parameters that can be changed
seed = 4200
n_train = 216
n_query = 1000
n_rounds = 5
dataset = "Xray"
strat =  ["Random", "Uncertainty", "Margin", "QBC"] #tells which strategy to use

with open(f"AL_scripts/results/parameters_{dataset}.pkl", "wb") as file:
    pickle.dump({"seed":seed, "n_train":n_train, "n_query":n_query, "n_rounds":n_rounds, "dataset":dataset, "strategy":strat}, file)


if __name__ == "__main__":
    run_AL(n_train, n_query, n_rounds, dataset, strat, seed = seed)