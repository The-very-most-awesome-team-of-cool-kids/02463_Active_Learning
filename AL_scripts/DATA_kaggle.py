# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:37:11 2020

@author: mat05
"""

import numpy as np
#from torchvision import datasets
import zipfile

from matplotlib import image
from matplotlib import pyplot
import glob, os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image, ImageOps
from kaggle.api.kaggle_api_extended import KaggleApi



#%% RUN ONCE

# Dowload from kaggle (needs login)
def download_kaggle(path):
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path = path)
    api.dataset_list_files('paultimothymooney/chest-xray-pneumonia').files


def unzip_kaggle(file_name):
    # unzip into train, test and val

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


# access images from folders and save in numpy  
#Create thumbnails in order to resize
def resize(path, cl, size = (256, 256)):
    for infile in glob.glob(path + cl +"/*.jpeg"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        im.thumbnail(size)
        im.save(file + ".thumbnail", "JPEG")

def create_files(resize_size):
    #create files: 
    path_train = r'./Egne_filer/Train/chest_xray/train/' 
    path_test = r'./Egne_filer/Test/chest_xray/test/'
    path_val = r'./Egne_filer/Val/chest_xray/val/'  
    resize(path_train, "NORMAL", size = resize_size)
    resize(path_train, "PNEUMONIA", size = resize_size)
    resize(path_test, "NORMAL", size = resize_size)
    resize(path_test, "PNEUMONIA", size = resize_size)
    resize(path_val, "NORMAL", size = resize_size)
    resize(path_val, "PNEUMONIA", size = resize_size)
    
def data_from_kaggle(resize_size):
    if not os.path.exists(".\Egne_filer\chest-xray-pneumonia.zip"):
        download_kaggle('.\Egne_filer')

    unzip_kaggle(".\Egne_filer\chest-xray-pneumonia.zip")
    create_files((resize_size, resize_size))



##################################################################

#print images
#im2 = Image.fromarray(X1[3])
#im2.show()


##################################################################

#im_pth = r'./Egne_filer/Val/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.thumbnail'


