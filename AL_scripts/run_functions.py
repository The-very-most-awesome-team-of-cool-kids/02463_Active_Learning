import numpy as np
from dataloader import get_dataset, get_handler, get_args
from model import get_model
from query_strategies import RandomSampling, UncertaintySampling, MarginSampling, QBCSampling
import torch
import pickle
import time
# import random


def train_rounds(X_te, Y_te, n_rounds, n_query, n_train, ids_labeled, strategy, seed):
    # calculate accuracies for the rounds
    acc = np.zeros(n_rounds+1)
    for rd in range(n_rounds+1):
        if rd != 0: #in round 0 our model has not been trained and we can therefore not use our query strategies
            #query
            q_ids = strategy.query(n_query)
            ids_labeled[q_ids] = True

            #update which labels are id'ed
            strategy.update(ids_labeled)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        strategy.train()
        preds = strategy.predict(X_te, Y_te)
        acc[rd] = 1.0*(Y_te.to(dtype=torch.int64) == preds).sum().item() / len(Y_te)
        print(f"The testing accuracy for round {rd} with {n_train+rd*n_query} training samples was {round(acc[rd], 2)*100} %") #/{sum(ids_labeled)}

    return acc


def run_AL(n_train, n_query, n_rounds, dataset, strat, seed = 1):
    """
    Strategy options: "random", "uncertainty"
    """
    if isinstance(strat, list):
        for s in strat:
            run_AL(n_train, n_query, n_rounds, dataset, s, seed)
        return None
    # Set parameters
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    args = get_args(dataset)

    # load dataset
    X_tr, Y_tr, X_te, Y_te = get_dataset(dataset)
    # X_tr = X_tr[:10000]
    # Y_tr = Y_tr[:10000]


    # inialize experiment
    n_pool = len(Y_tr)


    # generate the labeled observation to be inialised
    ids_labeled = np.zeros(n_pool, dtype = bool) # array of booleans that tells if the observation has been labeled
    ind_unlabeled = np.arange(n_pool) # array with indices of the unlabeled data
    np.random.shuffle(ind_unlabeled)
    ids_labeled[ind_unlabeled[:n_train]] = True # set the first n_train observations to be labeled

    # load model
    model = get_model(dataset)
    handler = get_handler(dataset)

    # define strategy
    if strat.lower() == "random":
        strategy = RandomSampling(X_tr, Y_tr, ids_labeled, model, handler, args)
    elif strat.lower() == "uncertainty":
        strategy = UncertaintySampling(X_tr, Y_tr, ids_labeled, model, handler, args)
    elif strat.lower() == "margin":
        strategy = MarginSampling(X_tr, Y_tr, ids_labeled, model, handler, args)
    elif strat.lower() == "qbc":
        strategy = QBCSampling(X_tr, Y_tr, ids_labeled, model, handler, args)

    # print info
    print("-"*20+" INFO "+"-"*20)
    print(f"Number of data points in the labeled pool: {n_train}")
    print(f"Number of data points in the unlabeled pool: {n_pool-n_train}")
    print(f"Number of data points in the test pool: {len(Y_te)}")
    print("The used data set is ", dataset)
    print(f"The seed is set to {seed}")
    print("The strategy is ", strategy.get_name())
    print("-"*46)

    t0 = time.time()
    acc = train_rounds(X_te, Y_te, n_rounds, n_query, n_train, ids_labeled, strategy, seed)
    print("Done!")
    print(f"Time for training: {time.time()-t0}")
    # save results
    with open("AL_scripts/results/"+dataset+ "_" +strategy.get_name()+ "_accuracies" +".pkl", "wb") as file:
        pickle.dump(acc, file)
