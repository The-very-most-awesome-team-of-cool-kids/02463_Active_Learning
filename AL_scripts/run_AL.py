import numpy as np
from dataloader import get_dataset, get_handler, get_args
from model import get_model
from query_strategies import RandomSampling
import torch
import pickle


# Parameters that can be changed
seed = 1
n_train = 10000
n_query = 1000
n_rounds = 10
dataset = "CIFAR10"
strat = "random_sampling" #tells which strategy to use

def run():
    # Set parameters
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    args = get_args(dataset)

    # load dataset
    X_tr, Y_tr, X_te, Y_te = get_dataset(dataset)
    #n_unlabeled = np.shape(X_tr)[0]-n_train
    X_tr = X_tr[:40000]
    Y_tr = Y_tr[:40000]

    # inialize experiment
    n_pool = len(Y_tr)
    print(f"Number of data points in the labeled pool: {n_train}")
    print(f"Number of data points in the unlabeled pool: {n_pool-n_train}")
    print(f"Number of data points in the test pool: {len(Y_te)}")

    # generate the labeled observation to be inialised
    ids_labeled = np.zeros(n_pool, dtype = bool) # array of booleans that tells if the observation has been labeled
    ind_unlabeled = np.arange(n_pool) # array with indices of the unlabeled data
    np.random.shuffle(ind_unlabeled)
    ids_labeled[ind_unlabeled[:n_train]] = True # set the first n_train observations to be labeled

    # load model
    model = get_model(dataset)
    handler = get_handler(dataset)

    if strat == "random_sampling":
        strategy = RandomSampling(X_tr, Y_tr, ids_labeled, model, handler, args)

    # print info
    print(dataset)
    print(f"Seed {seed}")
    print(type(strategy).__name__)

    # calculate accuracies for the rounds
    acc = np.zeros(n_rounds+1)
    for rd in range(n_rounds+1):
        if rd != 0: #in round 0 our model has not been trained and we can therefore not use our query strategies
            #query
            q_ids = strategy.query(n_query)
            ids_labeled[q_ids] = True

            #update which labels are id'ed
            strategy.update(ids_labeled)

        strategy.train()
        preds = strategy.predict(X_te, Y_te)
        acc[rd] = 1.0*(Y_te.to(dtype=torch.int64) == preds).sum().item() / len(Y_te)
        print(f"The testing accuracy for round {rd} was {acc[rd]}")

    # save results
    with open("AL_scripts/accuracies"+strat+".pkl", "wb") as file:
        pickle.dump(acc, file)

if __name__ == "__main__":
    run()