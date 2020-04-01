import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
# from DATA_kaggle import *

def get_dataset(name):
    """
    Gets data set:
    --------------------------------------
    Parameters:
    
    name: The name of the wanted data set, options are: "CIFAR10"
    --------------------------------------
    Outputs:

    X_tr: The training data
    Y_tr: The training labels
    X_te: The test data
    Y_te: The test labels
    """

    if name.upper() == "CIFAR10":
        data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
        data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
        X_tr = data_tr.train_data
        Y_tr = torch.from_numpy(np.array(data_tr.train_labels))
        X_te = data_te.test_data
        Y_te = torch.from_numpy(np.array(data_te.test_labels))
    elif name.upper() == "XRAY":
        path_train = r'./Egne_filer/Train/chest_xray/train/' 
        path_test = r'./Egne_filer/Test/chest_xray/test/'
        X_tr, y_tr = concat_(path_train)
        X_te, y_te = concat_(path_test)
        Y_tr = torch.from_numpy(y_tr)
        Y_te = torch.from_numpy(y_te)
    
    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name.upper() == "CIFAR10":
        return handler1
    elif name.upper() == "XRAY":
        return handler1
        

def get_args(name):
    if name.upper() == "CIFAR10":
        return {'n_epoch': 1,
                'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                'loader_tr_args':{'batch_size': 4, 'num_workers': 1},
                'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                'optimizer_args':{'lr': 0.0009}}
    if name.upper() == "XRAY":
        return {'n_epoch': 1,
                'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                'loader_tr_args':{'batch_size': 4, 'num_workers': 1},
                'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                'optimizer_args':{'lr': 0.0009}}



class handler1(Dataset):
    def __init__(self, X, Y, transform = None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
            return x, y, index

    def __len__(self):
        return len(self.X)

