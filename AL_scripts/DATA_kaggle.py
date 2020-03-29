# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:37:11 2020

@author: mat05
"""

import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path = '.\Egne_filer')
api.dataset_list_files('paultimothymooney/chest-xray-pneumonia').files


import io, pygame, zipfile
from zipfile import ZipFile 
zip_path = ".\Egne_filer\chest-xray-pneumonia.zip"
zf = zipfile.ZipFile(zip_path)
# read the images of zip via dataloader
train_loader = torch.utils.data.DataLoader(
                   DataSet(zf, transform),
                   batch_size = args.batch_size,
                   shuffle = True,
                   num_workers = args.workers,
                   pin_memory=True)
  
# specifying the zip file name 
file_name = ".\Egne_filer\chest-xray-pneumonia.zip"
  
# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall() 
    print('Done!') 
    


import zipfile
from StringIO import StringIO
from PIL import Image
import imghdr

imgzip = open('100-Test.zip')
zippedImgs = zipfile.ZipFile(imgzip)

