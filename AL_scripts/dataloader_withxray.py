import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from LOAD_XRAY import concat_, zeropad, Dataload as concat_, zeropad, Dataload

def get_dataset(name):
    if name.upper() == 'CIFAR10':
        return get_CIFAR10()
    elif name.upper() == 'XRAY':
        return get_Xray()

def get_CIFAR10():
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
    #if name.upper() == "CIFAR10":
    data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
    X_tr = data_tr.train_data
    Y_tr = torch.from_numpy(np.array(data_tr.train_labels))
    X_te = data_te.test_data
    Y_te = torch.from_numpy(np.array(data_te.test_labels))
    return X_tr, Y_tr, X_te, Y_te

    
    
   # elif name.upper() == "Xray":
def get_Xray():     
    path_train ="/Users/mat05/OneDrive - Danmarks Tekniske Universitet/02463_Active_Learning/AL_scripts/Egne_filer/Train/chest_xray/train/"
    path_test = "/Users/mat05/OneDrive - Danmarks Tekniske Universitet/02463_Active_Learning/AL_scripts/Egne_filer/Test/chest_xray/test/"
    X0_tr, y0_tr = Dataload(path_train, "NORMAL", 125)
    X1_tr, y1_tr = Dataload(path_train, "PNEUMONIA", 125)
    
    X_tr = np.concatenate((X0_tr,X1_tr),axis=0)   
    Y_tr = np.concatenate((y0_tr,y1_tr))

    X0_te, y0_te = Dataload(path_test, "NORMAL", 125)
    X1_te, y1_te = Dataload(path_test, "PNEUMONIA", 125)
    
    X_te = np.concatenate((X0_te,X1_te),axis=0)   
    Y_te = np.concatenate((y0_te,y1_te))
    
    Y_tr = torch.from_numpy(Y_tr)
    Y_te = torch.from_numpy(Y_te)
    
    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name.upper() == "CIFAR10":
        return handler1
    if name.upper() == "Xray":
        return handler1
        

def get_args(name):
    if name.upper() == "CIFAR10":
        return {'n_epoch': 1,
                'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                'loader_tr_args':{'batch_size': 4, 'num_workers': 1},
                'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                'optimizer_args':{'lr': 0.0009}}
        
        
    if name.upper() == "Xray":
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

