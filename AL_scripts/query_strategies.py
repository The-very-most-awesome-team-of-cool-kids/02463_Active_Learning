import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


class strategy:
    def __init__(self, X, Y, ids_labeled, model, handler, args):
        self.X = X
        self.Y = Y
        self.ids_labeled = ids_labeled
        self.model = model
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def query(self, n):
        pass

    def update(self, ids_labeled):
        """ Updates which ids have been labeled"""
        self.ids_labeled = ids_labeled


    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()

        for batch_id, (x, y, ids) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)
            optimizer.zero_grad()
            #print("shape of output: ",np.shape(self.clf(x)), self.clf(x))
            # out, e1 = self.clf(x)
            out = self.clf(x.float())
            # loss = nn.CrossEntropyLoss(out, y)
            # print(y, type(y))
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    def train(self):
        n_epoch = self.args['n_epoch']
        self.clf = self.model().to(self.device).float()
        optimizer = optim.Adam(self.clf.parameters(), **self.args['optimizer_args'])
        # print(self.X[self.ids_labeled], type(self.X[self.ids_labeled]))
        train_ids = np.arange(self.n_pool)[self.ids_labeled]
        loader_tr = DataLoader(self.handler(self.X[train_ids], self.Y[train_ids], transform = self.args["transform"]),
                                shuffle = True, **self.args["loader_tr_args"]) 
        
        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)

    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform = self.args["transform"]),
                                shuffle = True, **self.args["loader_te_args"])
        
        self.clf.eval()
        P = torch.zeros(len(Y), dtype = torch.int64)
        with torch.no_grad():
            for x, y, ids in loader_te:
                x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)
                # out, e1 = self.clf(x)
                out = self.clf(x)
                pred = out.max(1)[1]
                P[ids] = pred.cpu()

        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform = self.args["transform"]),
                                shuffle = True, **self.args["loader_te_args"])
        
        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, ids in loader_te:
                x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)
                # out, e1 = self.clf(x)
                out = self.clf(x)
                prob = F.softmax(out, dim = 1)
                probs[ids] = prob.cpu()

        return probs



class RandomSampling(strategy):
    def __init__(self, X, Y, ids_labeled, model, handler, args):
        super(RandomSampling, self).__init__(X, Y, ids_labeled, model, handler, args)

    def query(self, n):
        return np.random.choice(np.where(self.ids_labeled == 0)[0], n)

    def get_name(self):
        # return "RandomSampling"
        return type(self).__name__


class UncertaintySampling(strategy):
    def __init__(self, X, Y, ids_labeled, model, handler, args):
        self.uncertainty_measure = "Entropy"
        super(UncertaintySampling, self).__init__(X, Y, ids_labeled, model, handler, args)

    def set_uncertainty_measure(self, measure):
        """
        measure: the wanted uncertainty measure
        """
        self.uncertainty_measure = measure

    def get_name(self):
        return type(self).__name__


    def query(self, n):
        ids_unlabeled = np.arange(self.n_pool)[~self.ids_labeled]
        probs = self.predict_prob(self.X[ids_unlabeled], self.Y[ids_unlabeled])
        log_probs = torch.log(probs)
        if self.uncertainty_measure.lower() == "entropy":
            uncertainties = (-probs*log_probs).sum(1)
        
        return ids_unlabeled[uncertainties.sort(descending = True)[1][:n]]

class MarginSampling(strategy):
    def __init__(self, X, Y, ids_labeled, model, handler, args):
        super(MarginSampling, self).__init__(X, Y, ids_labeled, model, handler, args)

    def query(self, n):
        ids_unlabeled = np.arange(self.n_pool)[~self.ids_labeled]
        probs = self.predict_prob(self.X[ids_unlabeled], self.Y[ids_unlabeled])
        probs_sorted, ids = probs.sort(descending=True)
        margin = probs_sorted[:, 0] - probs_sorted[:,1]
        return ids_unlabeled[margin.sort()[1][:n]]

    def get_name(self):
        return type(self).__name__

class QBCSampling(strategy):
    def __init__(self, X, Y, ids_labeled, model, handler, args, n_committees = 10):
        self.n_committees = n_committees
        super(QBCSampling, self).__init__(X, Y, ids_labeled, model, handler, args)
    
    def query(self, n):
        ids_unlabeled = np.arange(self.n_pool)[~self.ids_labeled]
        preds = torch.zeros([len(ids_unlabeled), self.n_committees])

        # predict with different committees via bootstrapping
        for i in range(self.n_committees):
            strategy.update(self, ids_labeled = np.random.choice(self.ids_labeled, size = len(self.ids_labeled), replace = True))
            strategy.train(self)
            preds[:,i] = strategy.predict(self, self.X[ids_unlabeled], self.Y[ids_unlabeled])
        
        # use margin to find least confident
        classes = len(np.unique(self.Y))
        preds_prob = torch.tensor([np.histogram(preds[i, :], bins = np.arange(classes+1))[0]/classes for i in range(len(ids_unlabeled))])
        preds_margin, _ = preds_prob.sort(descending=True)
        preds_margin = preds_margin[:, 0] - preds_margin[:, 1]

        # set the indices with labels back to original
        strategy.update(self, ids_labeled= self.ids_labeled)
        return ids_unlabeled[preds_margin.sort()[1][:n]]
        

    def get_name(self):
        return type(self).__name__
