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



#%% RUN ONCE

# Dowload from kaggle (needs login)

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path = '.\Egne_filer')
api.dataset_list_files('paultimothymooney/chest-xray-pneumonia').files


# unzip into train, test and val

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



# access images from folders and save in numpy  
#Create thumbnails in order to resize
def resize(path, cl, size = (500,500)):
    for infile in glob.glob(path + cl +"/*.jpeg"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        im.thumbnail(size)
        im.save(file + ".thumbnail", "JPEG")

#create files: 
path_train = r'./Egne_filer/Train/chest_xray/train/' 
path_test = r'./Egne_filer/Test/chest_xray/test/'
path_val = r'./Egne_filer/Val/chest_xray/val/'  
resize(path_train, "NORMAL")
resize(path_train, "PNEUMONIA")
resize(path_test, "NORMAL")
resize(path_test, "PNEUMONIA")
resize(path_val, "NORMAL")
resize(path_val, "PNEUMONIA")
    




##################################################################

#print images
#im2 = Image.fromarray(X1[3])
#im2.show()


##################################################################

#im_pth = r'./Egne_filer/Val/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.thumbnail'


