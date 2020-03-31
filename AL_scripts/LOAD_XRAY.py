# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:24:14 2020

@author: mat05
"""
import numpy as np
#from torchvision import datasets


from matplotlib import image
from matplotlib import pyplot

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image, ImageOps

#%%
def zeropad(im_pth, N):
    #rturns zerropadded image
    im = Image.open(im_pth)
    old_size = im.size  
    
    new_size = tuple([int(x) for x in old_size])
    
    im = im.resize(new_size, Image.ANTIALIAS)
    
    new_im = Image.new("RGB", (N, N))
    new_im.paste(im, ((N-new_size[0])//2,
                          (N-new_size[1])//2))
    
    delta_w = N - new_size[0]
    delta_h = N - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    im = ImageOps.expand(im, padding, fill="black")
    data = np.asarray(im)
    
    return data

    
def get_Xray(path, cl, N):    
    #Normal images    
    all_files = glob.glob(path + cl + "/*.thumbnail")
    X = np.zeros((len(all_files),N,N))
    if cl == "NORMAL":
        clnum = 0
    elif cl== "PNEUMONIA":
        clnum = 1 
    y = np.asarray([clnum]*len(all_files))

    for i, filename in enumerate(all_files): 
        data = zeropad(filename,N)
        if len(data.shape) == 3: 
            pass
        else: 
            X[i,:,:] = data
    return X, y

def concat_(path,cl0="NORMAL",cl1="PNEUMONIA",N=500):
    
    X0, y0 = get_Xray(path, cl0, N)
    X1, y1 = get_Xray(path, cl1, N)
    
    X = np.concatenate((X0,X1),axis=0)   
    y = np.concatenate((y0,y1))
    return X, y
