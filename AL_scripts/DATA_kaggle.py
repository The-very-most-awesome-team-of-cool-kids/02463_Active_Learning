# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:37:11 2020

@author: mat05
"""

import numpy as np
#from torchvision import datasets
import zipfile
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


#%% Dowload from kaggle (needs login)

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path = '.\Egne_filer')
api.dataset_list_files('paultimothymooney/chest-xray-pneumonia').files


#%% unzip into train, test and val

file_name = ".\Egne_filer\chest-xray-pneumonia.zip"

archive = zipfile.ZipFile(file_name)
save_path = '.\Egne_filer\Train'
for file in archive.namelist():
    if file.startswith('chest_xray/train/'):
        archive.extract(file, save_path)
        
archive = zipfile.ZipFile(file_name)
save_path = '.\Egne_filer\Val'
for file in archive.namelist():
    if file.startswith('chest_xray/val/'):
        archive.extract(file, save_path)
        
archive = zipfile.ZipFile(file_name)
save_path = '.\Egne_filer\Test'
for file in archive.namelist():
    if file.startswith('chest_xray/test/'):
        archive.extract(file, save_path)



#%% access images from folders and save in numpy 
#shape of array
#PNEUMONIA = 1
#Normal = 0
M1, N1 = 1858,2090
path_train = r'./Egne_filer/Train/chest_xray/train/' 
path_test = r'./Egne_filer/Test/chest_xray/test/' 
def get_Xray(path_train, path_test, M1, N1):     
#path: path containing the folder "NORMAL" or "PNEUMONIA" 

    #TRAINING IMAGES
    #Normal images    
    all_files = glob.glob(path_train + "NORMAL" +"/*.jpeg")
    val0 = np.zeros((len(all_files),M1,N1))
    y0 = np.asarray([0]*len(all_files))
    for i, filename in enumerate(all_files):
        im = Image.open(filename)
        data = np.asarray(im)
        M,N = data.shape
        M2 = int((M1 - M)/2)
        N2 = int((N1 - N)/2)
        
        val0[i,M2:M+M2,N2:N+N2] = data
        
    #Pneumonia images
    all_files = glob.glob(path_train + "PNEUMONIA" +"/*.jpeg")
    val1 = np.zeros((len(all_files),M1,N1))
    y1 = np.asarray([1]*len(all_files))
    for j, filename in enumerate(all_files):
        im = Image.open(filename)
        data = np.asarray(im)
        M,N = data.shape
        M2 = int((M1 - M)/2)
        N2 = int((N1 - N)/2)
        
        val1[j,M2:M+M2,N2:N+N2] = data
        
    #Concatenate   
    X_tr = np.concatenate((val0,val1),axis=0)   
    Y_tr = np.concatenate((y0,y1))
    
    
    #Normal images    
    all_files = glob.glob(path_test + "NORMAL" +"/*.jpeg")
    val0 = np.zeros((len(all_files),M1,N1))
    y0 = np.asarray([0]*len(all_files))
    for i, filename in enumerate(all_files):
        im = Image.open(filename)
        data = np.asarray(im)
        M,N = data.shape
        M2 = int((M1 - M)/2)
        N2 = int((N1 - N)/2)
        
        val0[i,M2:M+M2,N2:N+N2] = data
        
    #Pneumonia images
    all_files = glob.glob(path_test + "PNEUMONIA" +"/*.jpeg")
    val1 = np.zeros((len(all_files),M1,N1))
    y1 = np.asarray([1]*len(all_files))
    for j, filename in enumerate(all_files):
        im = Image.open(filename)
        data = np.asarray(im)
        M,N = data.shape
        M2 = int((M1 - M)/2)
        N2 = int((N1 - N)/2)
        
        val1[j,M2:M+M2,N2:N+N2] = data
    #concatenate
    
    X_te = np.concatenate((val0,val1),axis=0)   
    Y_te = np.concatenate((y0,y1))
    
    return X_tr, Y_tr, X_te, Y_te

X_tr, Y_tr, X_te, Y_te = get_Xray(path_train, path_test, M1, N1)






#print images
im2 = Image.fromarray(val[0])
im2.show()
