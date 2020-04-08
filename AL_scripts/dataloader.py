import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from LOAD_XRAY import concat_, zeropad, Dataload as concat_, zeropad, Dataload
import os


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
        size = 256
        if not os.path.exists("Egne_filer/Train/chest_xray/"):
            from DATA_kaggle import data_from_kaggle
            print("Preparing data from Kaggle...")
            data_from_kaggle(size)
        path_train ="Egne_filer/Train/chest_xray/train/"
        path_test = "Egne_filer/Test/chest_xray/test/"
        X0_tr, y0_tr = Dataload(path_train, "NORMAL", size)
        X1_tr, y1_tr = Dataload(path_train, "PNEUMONIA", size)
        
        X_tr = np.concatenate((X0_tr,X1_tr),axis=0)   
        Y_tr = np.concatenate((y0_tr,y1_tr))

        X0_te, y0_te = Dataload(path_test, "NORMAL", size)
        X1_te, y1_te = Dataload(path_test, "PNEUMONIA", size)
        
        X_te = np.concatenate((X0_te,X1_te),axis=0)   
        Y_te = np.concatenate((y0_te,y1_te))
        
        Y_tr = torch.from_numpy(Y_tr)
        Y_te = torch.from_numpy(Y_te)
    
    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name.upper() == "CIFAR10":
        return handler1
    elif name.upper() == "XRAY":
        return handler2
        

def get_args(name):
    if name.upper() == "CIFAR10":
        return {'n_epoch': 1,
                'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                'loader_tr_args':{'batch_size': 4, 'num_workers': 1},
                'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                'optimizer_args':{'lr': 0.0009}}
    if name.upper() == "XRAY":
        return {'n_epoch': 1,
                'transform': transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 
                                # transforms.Resize(size=256),
                                # transforms.CenterCrop(size=224)]),
                'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                'loader_te_args':{'batch_size': 78, 'num_workers': 1},
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

class handler2(Dataset):
    def __init__(self, X, Y, transform = None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            h, w = np.shape(x)
            # print(type(x), np.shape(x))
            x = np.reshape(x, (h, w, 1))
            # print(type(x), np.shape(x))
            # # x = torch.FloatTensor(np.shape(x))
            # # x = transforms.functional.to_pil_image(x)
            # # x = Image.fromarray((x * 255).astype(np.uint8))
            # x = Image.fromarray(x)
            x = self.transform(x)
            return x, y, index

    def __len__(self):
        return len(self.X)